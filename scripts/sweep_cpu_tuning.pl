#!/usr/bin/env perl
# Sweep CPU Bit_DB set-operation tuning parameters and write LLM-friendly reports.
#
# Examples:
#   ./scripts/sweep_cpu_tuning.pl
#   ELEVATE=always CORES=0-9 REPS=5 PERF_REPS=5 ./scripts/sweep_cpu_tuning.pl
#   LIBPOPCNT_MODES=0,1 CPU_TILES=4,8,16 K_BLOCKS=256,512,1024 \
#     ./scripts/sweep_cpu_tuning.pl
#
# All comma-separated sweep variables may be overridden through the environment.
# The default is intentionally a direct-SIMD sweep (LIBPOPCNT=0); include mode 1
# explicitly when comparing the independent libpopcnt algorithm.

use strict;
use warnings;
use Cwd qw(getcwd abs_path);
use File::Path qw(make_path remove_tree);
use File::Spec;
use POSIX qw(strftime);

my $root = abs_path(File::Spec->catdir(File::Spec->curdir()));
die "Run this script from the repository root (Makefile not found).\n"
    unless -f File::Spec->catfile($root, 'Makefile');

sub csv_values {
    my ($name, $default) = @_;
    my $value = $ENV{$name} // $default;
    my @values = grep { length } split /\s*,\s*/, $value;
    die "$name must contain at least one comma-separated value\n" unless @values;
    return @values;
}

my @libpopcnt_modes = csv_values('LIBPOPCNT_MODES', '0');
my @cpu_tiles      = csv_values('CPU_TILES',       '4,8,16,32');
my @k_blocks       = csv_values('K_BLOCKS',        '256,512,768,1024');
my @shapes         = csv_values('SHAPES',          '1x1,2x2,2x4,4x2');
my @unrolls        = csv_values('UNROLLS',         '1,2,4');
my @buffer_sizes   = csv_values('BUFFER_SIZES',    '16,32,64,128');

my $cc          = $ENV{CC}          // 'clang';
my $cores       = $ENV{CORES}       // '0-9';
my $bit_length  = $ENV{BITS}        // 65536;
my $left_count  = $ENV{LEFT}        // 10240;
my $right_count = $ENV{RIGHT}       // 1024;
my $threads     = $ENV{THREADS}     // 10;
my $reps        = $ENV{REPS}        // 5;
my $perf_reps   = $ENV{PERF_REPS}   // 3;
my $elevate     = lc($ENV{ELEVATE}  // 'auto'); # never, auto, always
my $priority    = lc($ENV{PRIORITY} // 'nice'); # normal, nice, rr
my $limit       = $ENV{MAX_CONFIGS} // 0;
my $timestamp   = strftime('%Y%m%d-%H%M%S', localtime);
# Keep reports outside build/: every configuration invokes `make distclean`.
my $out_dir     = $ENV{OUT_DIR} // File::Spec->catdir($root, 'tuning-results', "tuning-$timestamp");
my $perf_events = $ENV{PERF_EVENTS} //
    'cycles,instructions,branches,branch-misses,cache-references,cache-misses';

for my $name (qw(BITS LEFT THREADS REPS PERF_REPS)) {
    my $value = {
        BITS => $bit_length, LEFT => $left_count, THREADS => $threads,
        REPS => $reps, PERF_REPS => $perf_reps,
    }->{$name};
    die "$name must be a positive integer\n" unless $value =~ /^\d+$/ && $value > 0;
}
die "RIGHT must be a positive integer\n" unless $right_count =~ /^\d+$/ && $right_count > 0;
die "ELEVATE must be never, auto, or always\n" unless $elevate =~ /^(?:never|auto|always)$/;
die "PRIORITY must be normal, nice, or rr\n" unless $priority =~ /^(?:normal|nice|rr)$/;

make_path($out_dir);

my $sudo = '';
if ($elevate ne 'never') {
    my $available = system('sudo -n true >/dev/null 2>&1') == 0;
    die "ELEVATE=always requires passwordless sudo (sudo -n).\n"
        if $elevate eq 'always' && !$available;
    $sudo = 'sudo -n ' if $available;
}
warn "Warning: running without sudo; perf or elevated priority may be unavailable.\n"
    if !$sudo && $elevate ne 'never';

sub shell_quote {
    return join ' ', map {
        my $arg = $_;
        $arg =~ s/'/'"'"'/g;
        "'$arg'";
    } @_;
}

sub run_command {
    my ($command, $log_path) = @_;
    my $rc = system("cd " . shell_quote($root) . " && $command > " . shell_quote($log_path) . " 2>&1");
    return $rc == 0;
}

sub parse_perf {
    my ($path) = @_;
    my %metrics;
    open my $fh, '<', $path or return \%metrics;
    while (my $line = <$fh>) {
        chomp $line;
        my @fields = split /,/, $line;
        next unless @fields >= 3;
        my ($value, undef, $event) = @fields[0, 1, 2];
        $value =~ s/\s+//g;
        $event =~ s/^\s+|\s+$//g;
        next if $value =~ /^(?:<notcounted>|<notsupported>|)$/;
        next unless $value =~ /^[-+]?\d+(?:\.\d+)?$/;
        $metrics{$event} = $value;
    }
    close $fh;
    return \%metrics;
}

sub parse_benchmark {
    my ($path) = @_;
    my %values;
    open my $fh, '<', $path or return \%values;
    while (my $line = <$fh>) {
        $values{best_ns} = $1 if $line =~ /^best:\s+(\d+)\s+ns/;
        $values{avg_ns} = $1 if $line =~ /^average:\s+([\d.]+)\s+ns/;
        $values{gqps} = $1 if $line =~ /^average:.*\(([\d.]+)\s+Gqword-pairs\/s\)/;
        $values{checksum} = $1 if $line =~ /^checksum:\s+(\d+)/;
    }
    close $fh;
    return \%values;
}

sub number {
    my ($value) = @_;
    return '' unless defined $value;
    $value =~ s/,//g;
    return $value + 0;
}

my @results;
my $index = 0;
for my $lib (@libpopcnt_modes) {
    die "LIBPOPCNT_MODES values must be 0 or 1\n" unless $lib =~ /^[01]$/;
    my @mode_unrolls = $lib ? ('-') : @unrolls;
    my @mode_buffers = $lib ? @buffer_sizes : ('-');

    for my $tile (@cpu_tiles) {
        for my $k_block (@k_blocks) {
            for my $shape (@shapes) {
                my ($rows, $cols) = $shape =~ /^(\d+)x(\d+)$/;
                die "Invalid SHAPES value '$shape'; use ROWSxCOLS, e.g. 2x4\n"
                    unless defined $rows && $rows > 0 && $cols > 0;
                for my $unroll (@mode_unrolls) {
                    for my $buffer (@mode_buffers) {
                        last if $limit && $index >= $limit;
                        ++$index;
                        my $tag = sprintf('%03d-lib%d-t%s-k%s-r%sc%s-u%s-b%s',
                            $index, $lib, $tile, $k_block, $rows, $cols, $unroll, $buffer);
                        print "[$index] $tag\n";

                        my @make_args = (
                            'make', 'distclean', '&&', 'make', 'GPU=NONE', "CC=$cc",
                            "CPU_TILE=$tile", "LIBPOPCNT=$lib",
                            "BITVECTOR_TILE=$k_block", "OUTER_ROW_NUM=$rows",
                            "OUTER_COL_NUM=$cols", 'bench_omp', 'APPLY_LTO=1',
                            'SIMD_DIAGNOSTICS=1',
                        );
                        push @make_args, "OUTER_VEC_BLK=$unroll" if !$lib;
                        push @make_args, "BUFFER_SIZE=$buffer" if $lib;
                        my $build_log = File::Spec->catfile($out_dir, "$tag.build.log");
                        my $built = run_command(join(' ', @make_args), $build_log);

                        my %result = (
                            tag => $tag, libpopcnt => $lib, cpu_tile => $tile,
                            k_block => $k_block, rows => $rows, cols => $cols,
                            unroll => $unroll, buffer_size => $buffer,
                            build_status => $built ? 'ok' : 'failed',
                            run_status => 'not-run',
                        );
                        if (!$built) {
                            push @results, \%result;
                            next;
                        }

                        my @priority_cmd;
                        if ($priority eq 'nice') {
                            @priority_cmd = ('nice', '-n', '-20');
                        } elsif ($priority eq 'rr') {
                            die "PRIORITY=rr requires ELEVATE=always or passwordless sudo\n" unless $sudo;
                            @priority_cmd = ('chrt', '-r', '50');
                        }
                        my $benchmark = File::Spec->catfile($root, 'build', 'openmp_bit_container');
                        my @run_args = (@priority_cmd, 'taskset', '-c', $cores, $benchmark,
                            $bit_length, $left_count, $right_count, $threads, $reps);
                        my $bench_log = File::Spec->catfile($out_dir, "$tag.benchmark.log");
                        my $perf_log = File::Spec->catfile($out_dir, "$tag.perf.csv");
                        my $command = join('', $sudo,
                            'env LC_ALL=C perf stat -x, -r ', $perf_reps,
                            ' -e ', shell_quote($perf_events),
                            ' -o ', shell_quote($perf_log), ' -- ', shell_quote(@run_args));
                        my $ran = run_command($command, $bench_log);
                        $result{run_status} = $ran ? 'ok' : 'failed';

                        my $bench = parse_benchmark($bench_log);
                        my $perf = parse_perf($perf_log);
                        $result{$_} = $bench->{$_} for keys %$bench;
                        for my $event (keys %$perf) {
                            (my $key = $event) =~ s/[^A-Za-z0-9]+/_/g;
                            $result{"perf_$key"} = $perf->{$event};
                        }
                        if (defined $result{avg_ns} && $result{avg_ns} > 0) {
                            $result{gqps} //= (($left_count * $right_count * int(($bit_length + 63) / 64)) / $result{avg_ns});
                        }
                        push @results, \%result;
                    }
                }
            }
        }
    }
}

my @columns = qw(tag build_status run_status libpopcnt cpu_tile k_block rows cols unroll buffer_size best_ns avg_ns gqps checksum perf_cycles perf_instructions perf_branches perf_branch_misses perf_cache_references perf_cache_misses);
my $csv_path = File::Spec->catfile($out_dir, 'summary.csv');
open my $csv, '>', $csv_path or die "Cannot write $csv_path: $!\n";
print {$csv} join(',', @columns), "\n";
for my $row (@results) {
    print {$csv} join(',', map { defined $row->{$_} ? $row->{$_} : '' } @columns), "\n";
}
close $csv;

my @valid = sort { number($a->{avg_ns}) <=> number($b->{avg_ns}) }
    grep { $_->{build_status} eq 'ok' && $_->{run_status} eq 'ok' && defined $_->{avg_ns} } @results;
my $report_path = File::Spec->catfile($out_dir, 'llm-summary.md');
open my $report, '>', $report_path or die "Cannot write $report_path: $!\n";
print {$report} "# Bit CPU tuning sweep\n\n";
print {$report} "## Measurement setup\n\n";
print {$report} "- CPU affinity: `$cores`\n";
print {$report} "- Priority: `$priority`; elevated execution: ", ($sudo ? 'yes' : 'no'), "\n";
print {$report} "- Benchmark: `openmp_bit_container $bit_length $left_count $right_count $threads $reps`\n";
print {$report} "- Perf repetitions per configuration: $perf_reps\n";
print {$report} "- Perf events: `$perf_events`\n";
print {$report} "- Configurations requested: ", scalar(@results), "\n";
print {$report} "- Successful configurations: ", scalar(@valid), "\n\n";
print {$report} "## Ranked successful configurations\n\n";
print {$report} "| Rank | Mode | CPU tile | K block | Shape | Unroll | Buffer | Average ns | Gqword-pairs/s | IPC | Cache miss % | Branch miss % |\n";
print {$report} "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|\n";
my $rank = 0;
for my $row (@valid) {
    ++$rank;
    my $cycles = number($row->{perf_cycles});
    my $instructions = number($row->{perf_instructions});
    my $cache_refs = number($row->{perf_cache_references});
    my $cache_misses = number($row->{perf_cache_misses});
    my $branches = number($row->{perf_branches});
    my $branch_misses = number($row->{perf_branch_misses});
    my $ipc = $cycles ? $instructions / $cycles : 0;
    my $cache_rate = $cache_refs ? 100 * $cache_misses / $cache_refs : 0;
    my $branch_rate = $branches ? 100 * $branch_misses / $branches : 0;
    printf {$report} "| %d | %d | %s | %s | %dx%d | %s | %s | %.0f | %.3f | %.3f | %.3f | %.3f |\n",
        $rank, $row->{libpopcnt}, $row->{cpu_tile}, $row->{k_block},
        $row->{rows}, $row->{cols}, $row->{unroll}, $row->{buffer_size},
        number($row->{avg_ns}), number($row->{gqps}), $ipc, $cache_rate, $branch_rate;
}
print {$report} "\n## Failed configurations\n\n";
my @failed = grep { $_->{build_status} ne 'ok' || $_->{run_status} ne 'ok' } @results;
print {$report} "- None\n" unless @failed;
print {$report} "- `$_->{tag}`: build=$_->{build_status}, run=$_->{run_status}\n" for @failed;
print {$report} "\n## Raw data\n\n- CSV: `summary.csv`\n- Per-configuration build logs: `*.build.log`\n- Per-configuration benchmark output: `*.benchmark.log`\n- Per-configuration perf output: `*.perf.csv`\n";
close $report;

print "\nCompleted ", scalar(@results), " configurations.\n";
print "LLM report: $report_path\n";
print "Raw CSV:    $csv_path\n";
