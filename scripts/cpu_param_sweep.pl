#!/usr/bin/env perl
use v5.36;
use File::Basename qw(dirname basename);
use File::Path     qw(make_path);
use File::Spec;
use Algorithm::Loops qw(NestedLoops);
use List::Util       qw(shuffle);
use Text::ParseWords qw(shellwords);
use Getopt::Long     qw(GetOptions);
use POSIX            qw(strftime);
use FindBin;

# Resolve the absolute path to the project root
my $root_dir = File::Spec->rel2abs( File::Spec->catdir( $FindBin::Bin, '..' ) );
my $exec_binary = File::Spec->catfile( $root_dir, 'build', 'openmp_bit_nogpu' );

my $summary_csv = '';
my $raw_log     = '';
my $output_dir  = File::Spec->catfile( $root_dir, 'benchmark_CPU_params' );
my $make_args   = '';
my $dry_run     = 0;
my $cores_list  = '1,10,20';

GetOptions(
    'summary|csv=s' => \$summary_csv,
    'log=s'         => \$raw_log,
    'out-dir=s'     => \$output_dir,
    'make-args=s'   => \$make_args,
    'cores=s'       => \$cores_list,
    'dry-run'       => \$dry_run,
) or die "Usage: $0 [--cores 1,10,20] [--make-args '...'] [--dry-run]\n";

my %target_cores = map { $_ => 1 } split( /\s*,\s*/, $cores_list );
my $max_threads  = 0;
for my $c ( keys %target_cores ) { $max_threads = $c if $c > $max_threads; }

my $detected_cpu = get_linux_cpu_name();
my ( $simd_level, $vpop_hw ) = detect_cpu_features();
my $has_numactl = ( system("command -v numactl >/dev/null 2>&1") == 0 );

my %parameters = (
    num_bits       => [  65536, 262144 ],
    libpopcnt      => [ 0,   1 ],
    cpu_tile       => [ 16,  32,   64 ],
    bitvector_tile => [ 512, 1024, 2048 ],
    buffer_size    => [ 128, 256,  512, 1024, 4096 ],
    outer_row_num  => [ 1,   2,    3,   4,    5, 6 ],
    outer_col_num  => [ 1,   2,    3,   4,    5, 6 ],
    outer_vec_blk  => [ 1,   2,    3,   4,    6, 8 ]
);

my @order = qw(num_bits libpopcnt cpu_tile bitvector_tile
  buffer_size outer_row_num outer_col_num outer_vec_blk
);

prepare_outputs();

my @benchmarks;
my @arrays = map { $parameters{$_} } @order;
NestedLoops( \@arrays,
    sub { my %current; @current{@order} = @_; push @benchmarks, \%current; } );
@benchmarks = shuffle(@benchmarks);

# Enforce strict thread affinity for NUMA architectures
$ENV{OMP_PROC_BIND} = 'close';
$ENV{OMP_PLACES}    = 'cores';

for my $opts (@benchmarks) { run_benchmark(%$opts); }

# --- Subroutines ---

sub detect_cpu_features {
    my $simd = 'None';
    my $vpop = 'None';
    if ( open my $fh, '<', '/proc/cpuinfo' ) {
        while ( my $line = <$fh> ) {
            if ( $line =~ /^flags/i || $line =~ /^Features/i ) {
                if ( $line =~ /avx512f/i ) {
                    $simd = 'AVX512';
                    $vpop = 'vpopcntdq' if $line =~ /avx512_vpopcntdq/i;
                }
                elsif ( $line =~ /avx2/i ) {
                    $simd = $simd eq 'AVX512' ? $simd : 'AVX2';
                }
                elsif ( $line =~ /avx/i ) {
                    $simd = $simd =~ /AVX/ ? $simd : 'AVX';
                }
                elsif ( $line =~ /sse4_2/i ) {
                    $simd = $simd ne 'None' ? $simd : 'SSE4.2';
                }
                elsif ( $line =~ /asimd/i ) {
                    $simd = 'NEON';
                    $vpop = 'vcntq_u8';
                }
            }
            elsif ( $line =~ /^isa/i ) {
                if ( $line =~ /v/i ) {
                    $simd = 'RISC-V Vector';
                    $vpop = 'vcpop.v' if $line =~ /zvbb/i;
                }
            }
        }
        close $fh;
    }
    return ( $simd, $vpop );
}

sub get_linux_cpu_name {
    my $cpu_name = 'Unknown_CPU';
    if ( open my $info, '<', '/proc/cpuinfo' ) {
        while ( my $line = <$info> ) {
            if ( $line =~ /^model name\s*:\s*(.+)$/ ) {
                $cpu_name = $1;
                $cpu_name =~ s/\(R\)|\(TM\)//gi;
                $cpu_name =~ s/^\s+|\s+$//g;
                $cpu_name =~ s/\s+/ /g;
                last;
            }
        }
        close $info;
    }
    return $cpu_name;
}

sub prepare_outputs {
    make_path($output_dir) unless -d $output_dir;

    if ( !$summary_csv || !$raw_log ) {
        my $clean_cpu = $detected_cpu;
        $clean_cpu =~ s/[^A-Za-z0-9]+/_/g;
        $clean_cpu =~ s/^_|_$//g;

        my $clean_simd = $simd_level;
        $clean_simd =~ s/[^A-Za-z0-9]+/_/g;

        my $timestamp = strftime( "%Y%m%d_%H%M%S", localtime );
        my $base_name = "cpu_sweep_${clean_cpu}_${clean_simd}_${timestamp}";

        $summary_csv = "${base_name}.csv"     unless $summary_csv;
        $raw_log     = "${base_name}_raw.log" unless $raw_log;
    }

    $summary_csv = File::Spec->catfile( $output_dir, $summary_csv )
      unless $summary_csv =~ m{[/\\]};
    $raw_log = File::Spec->catfile( $output_dir, $raw_log )
      unless $raw_log =~ m{[/\\]};

    open my $fh, '>', $summary_csv or die "Cannot create CSV: $!\n";
    say $fh "Processor,SIMD,vpopcountHW,LIBPOPCNT,Benchmark_Type"
      . ",Bitset_Size,Threads,CPU_TILE,BITVECTOR_TILE,"
      . "BUFFER_SIZE,OUTER_ROW_NUM,OUTER_COL_NUM,OUTER_VEC_BLK,Timing_ns";
    close $fh;

    log_message("Detected Processor: $detected_cpu\n");
    log_message("Output CSV initialized at: $summary_csv\n");
}

sub run_benchmark {
    my %opts = @_;

    return if $dry_run;

    # 1. Clean previous build artifacts quietly
    my $clean_cmd =
      "make -C " . quotemeta($root_dir) . " distclean > /dev/null 2>&1";
    system($clean_cmd);

    # 2. Build compile command array
    my @compile_cmd = (
        'make',
        '-C',
        $root_dir,
        '-f',
        'Makefile',
        '-B',
        'build/openmp_bit_nogpu',
        "GPU=NONE",
        "LIBPOPCNT=$opts{libpopcnt}",
        "CPU_TILE=$opts{cpu_tile}",
        "BITVECTOR_TILE=$opts{bitvector_tile}",
        "BUFFER_SIZE=$opts{buffer_size}",
        "OUTER_ROW_NUM=$opts{outer_row_num}",
        "OUTER_COL_NUM=$opts{outer_col_num}",
        "OUTER_VEC_BLK=$opts{outer_vec_blk}"
    );

    push @compile_cmd, shellwords($make_args) if $make_args;

    # Log the clean, unescaped command for tracking purposes
    log_message( "COMPILING: " . join( ' ', @compile_cmd ) . "\n" );

    # Safely package the array into a shell string with output redirection
    my $compile_str =
      join( ' ', map { quotemeta($_) } @compile_cmd ) . " > /dev/null 2>&1";
    if ( system($compile_str) != 0 ) {
        log_message("ERROR: Compilation failed.\n");
        return;
    }

    # 3. Execute benchmark binary (keep stdout unredirected for parsing)
    my $exec_cmd = "$exec_binary $opts{num_bits} 1024 1024 $max_threads";
    $exec_cmd = "numactl --interleave=all $exec_cmd" if $has_numactl;

    log_message("EXECUTING: $exec_cmd\n");
    my $output = `$exec_cmd 2>&1`;

    # Regex capturing optional "Container - " string to determine benchmark type
    my $bench_pat = qr{
        Total \s time \s for \s (?<type>Container \s - \s )? Multi-threaded \s - \s OpenMP:
        \s+ (?<timing> \d+ ) \s ns
        .*?
        Number \s of \s threads: \s+ (?<threads> \d+ )
    }x;

    open my $out, '>>', $summary_csv or die "Cannot append CSV: $!\n";
    for my $line ( split /\n/, $output ) {
        if ( $line =~ $bench_pat ) {
            if ( exists $target_cores{ $+{threads} } ) {
                my $b_type = $+{type} ? 'Containerized' : 'Non-Containerized';

                say {$out} join ',',
                  $detected_cpu,
                  $simd_level,
                  $vpop_hw,
                  $opts{libpopcnt},
                  $b_type,
                  $opts{num_bits}, $+{threads},
                  $opts{cpu_tile},
                  $opts{bitvector_tile},
                  $opts{buffer_size},
                  $opts{outer_row_num},
                  $opts{outer_col_num},
                  $opts{outer_vec_blk},
                  $+{timing};
            }
        }
    }
    close $out;
}

sub log_message {
    my ($message) = @_;
    open my $logfh, '>>', $raw_log or die "Cannot append to raw log: $!\n";
    print $logfh $message;
    close $logfh;
}
