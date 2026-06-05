#!/usr/bin/env perl
use strict;
use warnings;
use File::Basename qw(dirname basename);
use File::Path     qw(make_path);
use File::Temp     qw(tempfile);
use File::Spec;
use Algorithm::Loops qw(NestedLoops);
use List::Util qw(shuffle);
use Text::ParseWords qw(shellwords);
use Getopt::Long     qw(GetOptions);
use POSIX            qw(strftime);

my $summary_csv     = 'gpu_param_sweep.csv';
my $raw_log         = 'gpu_param_sweep_raw.log';
my $backend_list    = 'NVIDIA';
my $architecture    = '';
my $make_args       = '';
my $output_dir      = 'benchmark_GPU_params';
my $dry_run         = 0;
my $iterations_list = '10';

GetOptions(
    'summary|csv=s' => \$summary_csv,
    'log=s'         => \$raw_log,
    'out-dir=s'     => \$output_dir,
    'backend=s'     => \$backend_list,
    'arch=s'        => \$architecture,
    'make-args=s'   => \$make_args,
    'iterations=s'  => \$iterations_list,
    'dry-run'       => \$dry_run,
  )
  or die
"Usage: $0 [--summary file] [--log file] [--out-dir dir] [--backend comma,separated] [--make-args '...'] [--iterations list] [--repeat N] [--dry-run]\n";

set_default_output_paths();
normalize_output_paths();

my %parameters = (
    backend     => [ split( /\s*,\s*/, $backend_list ) ],
    tile_j      => [ 512, 1024, 2048, 4096 ],
    ilp         => [ 4,   8,    16,   32, 64 ],
    num_bits    => [ 16384, 65536, 262144 ],
    num_queries => [ 1000, 5000, 10000 ],
    num_refs    => [ 1024, 2048, 4096, 8192 ],
    iterations  => [ map { int($_) } split( /\s*,\s*/, $iterations_list ) ],
);

my @order = qw(backend tile_j ilp num_bits num_queries num_refs iterations);
main();

sub normalize_output_paths {
    $output_dir =~ s{[\\/]+$}{};
    $output_dir = File::Spec->canonpath($output_dir);

    if ( !path_has_directory($summary_csv) ) {
        $summary_csv = File::Spec->catfile( $output_dir, $summary_csv );
    }
    if ( !path_has_directory($raw_log) ) {
        $raw_log = File::Spec->catfile( $output_dir, $raw_log );
    }
}

sub path_has_directory {
    my ($path) = @_;
    my ( undef, $dirs, $file ) = File::Spec->splitpath($path);
    return defined $dirs && $dirs ne '' && $dirs ne '.';
}

sub set_default_output_paths {
    my @backends = split( /\s*,\s*/, $backend_list );
    my $backend  = $backends[0] // 'gpu';
    my $arch     = infer_architecture( $backend, $make_args );

    my $backend_label = lc($backend);
    $backend_label =~ s/[^a-z0-9]+/_/g;
    my $arch_label = lc($arch);
    $arch_label =~ s/[^a-z0-9]+/_/g;
    $arch_label = 'implicit' if $arch_label eq '';

    my $summary_name = "${backend_label}_${arch_label}_sweep_results.csv";
    my $raw_name     = "${backend_label}_${arch_label}_sweep_raw.log";
    if ( !path_has_directory($summary_csv) ) {
        $summary_csv = File::Spec->catfile( $output_dir, $summary_name );
    }
    if ( !path_has_directory($raw_log) ) {
        $raw_log = File::Spec->catfile( $output_dir, $raw_name );
    }
}

sub main {
    prepare_outputs();

    my $start_time = strftime( '%Y-%m-%d %H:%M:%S', localtime );
    log_message("Starting GPU parameter sweep at $start_time\n");

    my @benchmarks;
    my @arrays = map { $parameters{$_} } @order;
    NestedLoops(
        \@arrays,
        sub {
            my @values = @_;
            my %current;
            @current{@order} = @values;
            push @benchmarks, \%current;
        }
    );

    @benchmarks = shuffle(@benchmarks);
    for my $opts (@benchmarks) {
        run_benchmark(%$opts);
    }

    my $end_time = strftime( '%Y-%m-%d %H:%M:%S', localtime );
    log_message("Completed GPU parameter sweep at $end_time\n");
}

sub prepare_outputs {
    $output_dir = File::Spec->canonpath($output_dir);
    make_path($output_dir) if $output_dir ne '' && $output_dir ne '.';
    make_path( dirname($summary_csv) ) if dirname($summary_csv) ne '.';
    make_path( dirname($raw_log) )     if dirname($raw_log) ne '.';

    unlink $summary_csv if -e $summary_csv;
    unlink $raw_log     if -e $raw_log;

    open my $fh, '>', $summary_csv
      or die "Cannot create summary CSV '$summary_csv': $!\n";
    print $fh "backend,architecture,TILE_J,ILP,number of bits in bitsets,"
      . "number of queries,number of reference sequences,iteration count,"
      . "timing type,timing (ns),iterations per second\n";
    close $fh;
}

sub run_benchmark {
    my %opts = @_;
    my $arch = infer_architecture( $opts{backend}, $make_args );

    check_log_size();

    my ( $tmpfh, $tmpfile ) = tempfile(
        'gpu_csv_XXXX',
        SUFFIX => '.csv',
        DIR    => File::Spec->tmpdir(),
    );
    close $tmpfh;

    my ( $logfh, $tmplog ) = tempfile(
        'gpu_log_XXXX',
        SUFFIX => '.log',
        DIR    => File::Spec->tmpdir(),
    );
    close $logfh;

# 1. Define the command and arguments as an array
    my @command = (
        'make',
        '-B',
        "GPU=$opts{backend}",
        'gpu_bench_csv',
        "GPU_TILE_J=$opts{tile_j}",
        "GPU_ILP=$opts{ilp}",
        "GPU_NUM_BITS=$opts{num_bits}",
        "GPU_NUM_QUERIES=$opts{num_queries}",
        "GPU_NUM_REFS=$opts{num_refs}",
        "GPU_ITERATIONS=$opts{iterations}",
        "GPU_CSV_OUTPUT=$tmpfile",
        "GPU_LOG_OUTPUT=$tmplog"
    );

    # 2. Safely handle user-provided make_args by splitting them on spaces
    if ($make_args) {
        push @command, shellwords($make_args);
    }

    # 3. Log it for debugging (join it just for the log)
    log_message("RUN: " . join(' ', @command) . "\n");
    return if $dry_run;

    # 4. Execute safely bypassing the shell.
    # Remove any conflicting GPU architecture environment variables that
    # would otherwise affect make even when explicit make-args are provided.
    local $ENV{AMD_ARCH}   if $opts{backend} eq 'NVIDIA';
    local $ENV{NVIDIA_ARCH} if $opts{backend} eq 'AMD';
    my $rc = system(@command);
    
    if ( $rc != 0 ) {
        log_message("ERROR: benchmark failed with exit code " . ( $rc >> 8 ) . "\n");
        die "Benchmark command failed.\n";
    }

    append_tmp_csv( $tmpfile, $opts{backend}, $arch );
    append_tmp_log( $tmplog );
    unlink $tmpfile;
    unlink $tmplog;
}

sub append_tmp_log {
    my ($tmplog) = @_;
    open my $in, '<', $tmplog
      or die "Cannot open temporary log '$tmplog': $!\n";
    open my $out, '>>', $raw_log
      or die "Cannot append to raw log '$raw_log': $!\n";

    while ( my $line = <$in> ) {
        print $out $line;
    }

    close $in;
    close $out;
}

sub check_log_size {
    my $max_bytes = 10 * 1024 * 1024 * 1024;
    if ( -e $raw_log ) {
        my $size = -s $raw_log;
        if ( defined $size && $size > $max_bytes ) {
            die sprintf(
                "Raw log '%s' is too large (%.2f GB); stopping to avoid runaway logs.\n",
                $raw_log, $size / (1024 * 1024 * 1024)
            );
        }
    }
}

sub infer_architecture {
    my ( $backend, $make_args ) = @_;
    if ( $architecture && $architecture ne '' ) {
        return $architecture;
    }
    for my $token ( split /\s+/, $make_args // '' ) {
        if (   $backend eq 'NVIDIA'
            && $token =~ /^NVIDIA_ARCH=(?:'([^']*)'|"([^\"]*)"|(.+))$/ )
        {
            return defined $1 ? $1 : defined $2 ? $2 : $3;
        }
        if (   $backend eq 'AMD'
            && $token =~ /^AMD_ARCH=(?:'([^']*)'|"([^\"]*)"|(.+))$/ )
        {
            return defined $1 ? $1 : defined $2 ? $2 : $3;
        }
    }
    if ( $backend eq 'NVIDIA' ) {
        if ( $make_args =~
            /(?:^|\s)NVIDIA_ARCH\s*=\s*(?:'([^']*)'|"([^\"]*)"|([^\s]+))/ )
        {
            return defined $1 ? $1 : defined $2 ? $2 : $3;
        }
    }
    if ( $backend eq 'AMD' ) {
        if ( $make_args =~
            /(?:^|\s)AMD_ARCH\s*=\s*(?:'([^']*)'|"([^\"]*)"|([^\s]+))/ )
        {
            return defined $1 ? $1 : defined $2 ? $2 : $3;
        }
    }
    return 'implicit';
}

sub append_tmp_csv {
    my ( $tmpfile, $backend, $arch ) = @_;
    open my $in, '<', $tmpfile
      or die "Cannot open temporary CSV '$tmpfile': $!\n";
    open my $out, '>>', $summary_csv
      or die "Cannot append to summary CSV '$summary_csv': $!\n";

    my $header = <$in>;
    while ( my $line = <$in> ) {
        chomp $line;
        next unless length $line;
        print $out "$backend,$arch,$line\n";
    }

    close $in;
    close $out;
}

sub log_message {
    my ($message) = @_;
    open my $logfh, '>>', $raw_log
      or die "Cannot append to raw log '$raw_log': $!\n";
    print $logfh $message;
    close $logfh;
    print $message;
}

sub make_range {
    my ( $start, $end, $step ) = @_;
    my @values;
    for ( my $val = $start ; $val <= $end ; $val += $step ) {
        push @values, $val;
    }
    return @values;
}

sub shell_quote {
    my ($value) = @_;
    $value =~ s/'/'"'"'/g;
    return "'$value'";
}
