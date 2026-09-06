#!/usr/bin/env perl
# faiss_compare.pl -- Small FAISS comparison sweep orchestrator.
#
# Compares four "builds" across a Cartesian run grid and harvests
# PER-ITERATION timings (nanoseconds) into a long-format CSV:
#   faiss_cpu : scripts/faiss_cpu_benchmark.py   (native FAISS, host)
#   faiss_gpu : scripts/faiss_gpu_benchmark.py   (native FAISS, GPU 0)
#   bit_cpu   : build/openmp_bit_cpu_FAISS_comp  (Bit OpenMP, host top-k)
#   bit_gpu   : build/openmp_bit_gpu_FAISS_comp  (Bit OpenMP, device top-k)
#
# The orchestrator owns the entire Cartesian product; every target is a
# "dumb" single-configuration executor invoked once per grid cell. All
# parameters come from a single-source-of-truth JSON schema
# (scripts/benchmark_config_faiss.json) and can be overridden on the CLI.
#
# Usage:
#   perl scripts/faiss_compare.pl --config scripts/benchmark_config_faiss.json
#   perl scripts/faiss_compare.pl --config ... --bitset_bits 1024,4096 --dry_run
#
# Requires: JSON::PP (core), Getopt::Long (core), Text::ParseWords (core),
#           IPC::Run, Algorithm::Loops.

use strict;
use warnings;
use Getopt::Long ();
use JSON::PP qw(decode_json);
use Text::ParseWords qw(shellwords);
use IPC::Run ();
use Algorithm::Loops qw(NestedLoops);
use File::Path qw(make_path);
use Time::HiRes qw(gettimeofday tv_interval);

# ---------------------------------------------------------------------------
# Pass 1: intercept and strip the configuration flag, then decode the JSON.
# ---------------------------------------------------------------------------
Getopt::Long::Configure("pass_through");
my $config_file = "scripts/benchmark_config_faiss.json";
Getopt::Long::GetOptions("config=s" => \$config_file);

open my $cfh, '<', $config_file
  or die "ERROR: cannot open config '$config_file': $!\n";
my $config = decode_json(do { local $/; <$cfh> });
close $cfh;

my $sys          = $config->{system_env}    // {};
my $build_matrix = $config->{build_matrix}  // {};
my $run_matrix   = $config->{run_matrix}    // {};
my $parsers      = $config->{output_parser}{by_kind} // {};
my $build_cfg    = $config->{build}         // {};

# 'build.gpu' is required: it selects the offload target for the up-front build
# ('NONE' = CPU-only comparison; the GPU comparator/benchmarks are then dropped).
die "ERROR: config 'build.gpu' is required (set to 'NONE' for CPU-only).\n"
  unless defined $build_cfg->{gpu} && length $build_cfg->{gpu};

# ---------------------------------------------------------------------------
# Pass 2: dynamically bind the remaining CLI arguments from the JSON schema.
# Any system_env scalar and any run_matrix dimension can be overridden.
# ---------------------------------------------------------------------------
my %cli;
my @option_specs;
for my $key (sort keys %{$sys}) {
  push @option_specs, "$key=s";
}
for my $dim (sort keys %{$run_matrix}) {
  push @option_specs, "$dim=s";    # comma-separated list override
}
push @option_specs, "dry_run!", "help!";
Getopt::Long::Configure("no_pass_through");
Getopt::Long::GetOptions(\%cli, @option_specs);

if ($cli{help}) {
  print <<"USAGE";
Usage: perl scripts/faiss_compare.pl [--config FILE] [overrides] [--dry_run]
  Overrides (any system_env key or run_matrix dimension):
    --out_dir DIR            results directory      (default: $sys->{out_dir})
    --iterations N           measured iterations    (default: $sys->{iterations})
    --bitset_bits LIST       e.g. 1024,4096         (default: @{$run_matrix->{bitset_bits}})
    --top_k LIST             e.g. 64,256            (default: @{$run_matrix->{top_k}})
    --num_queries N          query bitsets          (default: $sys->{num_queries})
    --num_refs LIST          e.g. 10000,100000      (default: @{$run_matrix->{num_refs}})
    --threads N|auto         CPU threads            (default: $sys->{threads})
    --gpu_id N               GPU device             (default: $sys->{gpu_id})
    --dry_run                print commands without executing
USAGE
  exit 0;
}

# Merge overrides over the JSON defaults.
for my $key (keys %{$sys}) {
  $sys->{$key} = $cli{$key} if defined $cli{$key};
}
for my $dim (keys %{$run_matrix}) {
  $run_matrix->{$dim} = [ split /,/, $cli{$dim} ] if defined $cli{$dim};
}

my $iterations = $sys->{iterations};
die "ERROR: iterations must be >= 1\n" if $iterations < 1;
warn "NOTE: iterations=$iterations; >= 100 is recommended for distributions\n"
  if $iterations < 100;

# ---------------------------------------------------------------------------
# Environment resolution and preflight.
# ---------------------------------------------------------------------------
my $threads = $sys->{threads};
if ($threads eq 'auto') {
  chomp( my $nproc = `nproc 2>/dev/null` // '' );
  $threads = ( $nproc =~ /^\d+$/ && $nproc > 0 ) ? $nproc : 1;
}

# numactl is optional: degrade gracefully (warn and drop the prefix).
my $numactl = $sys->{numactl_prefix} // '';
if ( length $numactl ) {
  my ($numactl_bin) = shellwords($numactl);
  if ( system("command -v $numactl_bin >/dev/null 2>&1") != 0 ) {
    warn "WARNING: '$numactl_bin' not found; NUMA interleaving disabled\n";
    $numactl = '';
  }
}

# Resolve the Python interpreter for the FAISS builds. The OpenMP builds are
# independent of FAISS: only the two native FAISS builds use this interpreter.
# If the bare 'python' cannot import faiss, fall back to the configured conda
# environment (default: faiss_env) without touching the rest of the sweep.
my $python   = $sys->{python}   // 'python';
my $conda_env = $sys->{conda_env} // '';
sub python_has_faiss {
  my ($py) = @_;
  return system("$py -c \"import faiss\" >/dev/null 2>&1") == 0;
}
my $faiss_ok = python_has_faiss($python);
if ( !$faiss_ok && length $conda_env ) {
  my $candidate =
    "conda run -n $conda_env --no-capture-output python";
  if ( python_has_faiss($candidate) ) {
    $python   = $candidate;
    $faiss_ok = 1;
  }
}

# GPU preflight: FAISS must see at least the requested GPU id.
my $gpu_ok = 0;
if ($faiss_ok) {
  my $probe = "$python -c \"import faiss; print(faiss.get_num_gpus())\" 2>/dev/null";
  chomp( my $ngpu = `$probe` // '' );
  if ( $ngpu =~ /^(\d+)$/ && $1 > $sys->{gpu_id} ) {
    $gpu_ok = 1;
  }
}
if ( !$faiss_ok ) {
  warn "WARNING: no Python interpreter can import faiss "
     . "(tried '$sys->{python}'"
     . ( length $conda_env ? " and conda env '$conda_env'" : '' )
     . "); native FAISS builds (faiss_cpu, faiss_gpu) will be skipped. "
     . "The OpenMP builds are unaffected\n";
}
if ( !$gpu_ok ) {
  warn "WARNING: FAISS reports no usable GPU $sys->{gpu_id}; "
     . "GPU builds (faiss_gpu, bit_gpu) will be skipped\n";
}

# ---------------------------------------------------------------------------
# Up-front build: pin libbit and the comparators to a known, config-driven
# configuration. Runs ONCE before the grid; aborts the whole comparison on
# failure. Blank/empty config values are omitted so Makefile defaults apply.
# ---------------------------------------------------------------------------
sub run_build_step {
  my ($build_cfg) = @_;

  my $gpu = $build_cfg->{gpu};
  my $gpu_none = ( $gpu =~ /^none$/i ) ? 1 : 0;

  # Map lowercase config keys -> Make variables (host-build tunables).
  my %var_map = (
    cc                  => 'CC',
    cpu_tile            => 'CPU_TILE',
    bitvector_tile      => 'BITVECTOR_TILE',
    buffer_size         => 'BUFFER_SIZE',
    outer_row_num       => 'OUTER_ROW_NUM',
    outer_col_num       => 'OUTER_COL_NUM',
    outer_vec_blk       => 'OUTER_VEC_BLK',
    libpopcnt           => 'LIBPOPCNT',
    apply_lto           => 'APPLY_LTO',
    use_builtin_popcount => 'USE_BUILTIN_POPCOUNT',
  );

  # Blank-omit: only forward values that are defined and non-empty.
  my @make_vars = ( "GPU=$gpu" );
  my @effective = ( "GPU=$gpu" );
  if ( defined $build_cfg->{gpu_arch} && length $build_cfg->{gpu_arch} ) {
    push @make_vars, "GPU_ARCH=$build_cfg->{gpu_arch}";
    push @effective, "GPU_ARCH=$build_cfg->{gpu_arch}";
  }
  for my $key ( sort keys %var_map ) {
    my $val = $build_cfg->{$key};
    next unless defined $val;
    if ( ref($val) eq 'ARRAY' ) { next unless @$val; $val = join( ',', @$val ); }
    $val = trim_ws($val);
    next unless length $val;
    push @make_vars, "$var_map{$key}=$val";
    push @effective, "$var_map{$key}=$val";
  }

  my @targets = ('openmp_bit_cpu_FAISS_comp');
  push @targets, 'openmp_bit_gpu_FAISS_comp' if !$gpu_none;

  my @cmd = ( 'make', '-B', @targets, @make_vars );
  print "[build] ", join( ' ', @cmd ), "\n";
  return ( $gpu_none, \@effective ) if $cli{dry_run};

  my $build_out = '';
  # IPC::Run::run returns FALSE (does not die) on non-zero exit; capture that.
  my $ok = eval { IPC::Run::run( \@cmd, '>', \$build_out, '2>&1' ) };
  $ok = 0 if $@;    # IPC exception
  if ( !$ok ) {
    my @tail = split /\n/, $build_out;
    my $start = @tail > 40 ? $#tail - 39 : 0;
    die "ERROR: build failed (make exited non-zero). Last output:\n"
      . join( "\n", @tail[ $start .. $#tail ] ) . "\n";
  }
  return ( $gpu_none, \@effective );
}

sub trim_ws { my ($s) = @_; $s //= ''; $s =~ s/^\s+|\s+$//g; return $s; }

# The Bit executables are built by run_build_step() above; here we only verify
# presence and drop builds that cannot run. With gpu == NONE the GPU builds are
# dropped from the run matrix even if stale binaries are present.
my ( $gpu_none, $effective_build ) = run_build_step($build_cfg);
my %build_enabled;
for my $build ( sort keys %{$build_matrix} ) {
  my $spec = $build_matrix->{$build};
  my $enabled = 1;
  if ( $spec->{kind} eq 'faiss' && !$faiss_ok ) {
    $enabled = 0;
  }
  if ( $spec->{needs_gpu} && !$gpu_ok ) {
    $enabled = 0;
  }
  if ( $gpu_none && $spec->{needs_gpu} ) {
    warn "NOTE: build.gpu=NONE; dropping GPU build '$build' from the run\n";
    $enabled = 0;
  }
  if ( $spec->{kind} eq 'openmp' ) {
    my ($exe) = shellwords( $spec->{cmd} );
    if ( !-x $exe ) {
      warn "WARNING: '$exe' not found or not executable; build '$build' skipped\n";
      $enabled = 0;
    }
  }
  $build_enabled{$build} = $enabled;
}
my @builds = grep { $build_enabled{$_} } sort keys %{$build_matrix};
die "ERROR: no builds are runnable; nothing to do\n" unless @builds;

# ---------------------------------------------------------------------------
# Output setup.
# ---------------------------------------------------------------------------
my $out_dir     = $sys->{out_dir};
my $results_csv = "$out_dir/$sys->{results_csv}";
make_path($out_dir) if !$cli{dry_run};

# Self-describing sidecar: record the EFFECTIVE build configuration actually
# passed to make (only the non-omitted tunables), so each result set is
# reproducible without inspecting the JSON defaults.
if ( !$cli{dry_run} ) {
  if ( open my $bc, '>', "$out_dir/build_config.txt" ) {
    print {$bc} "# Effective build configuration (passed to 'make -B')\n";
    print {$bc} "$_\n" for @{$effective_build};
    close $bc;
  } else {
    warn "WARNING: cannot write '$out_dir/build_config.txt': $!\n";
  }
}

my @csv_header = qw(Backend Build Device Bitset_Bits Top_K Num_Queries Num_Refs
                    Threads Iteration Timing_ns);

my $out_fh;
if ( !$cli{dry_run} ) {
  open $out_fh, '>', $results_csv
    or die "ERROR: cannot write '$results_csv': $!\n";
  print {$out_fh} join( ',', @csv_header ), "\n";
}

# ---------------------------------------------------------------------------
# Command template expansion.
# ---------------------------------------------------------------------------
sub expand_cmd {
  my ( $template, %vars ) = @_;
  my $cmd = $template;
  for my $k ( keys %vars ) {
    my $v = $vars{$k};
    $cmd =~ s/\{$k\}/$v/g;
  }
  return $cmd;
}

# ---------------------------------------------------------------------------
# Per-iteration parser: apply the JSON regex for this build kind, return a
# list of [iteration, timing_ns] pairs.
# ---------------------------------------------------------------------------
sub parse_iterations {
  my ( $kind, $output ) = @_;
  my $spec = $parsers->{$kind} or die "ERROR: no output_parser for kind '$kind'\n";
  my $re   = qr/$spec->{regex}/m;
  my @rows;
  while ( $output =~ /$re/g ) {
    my $iter  = $+{Iteration};
    my $timing = $+{Timing_ns};
    next unless defined $iter && defined $timing;
    push @rows, [ $iter, $timing ];
  }
  return @rows;
}

# ---------------------------------------------------------------------------
# Cartesian sweep: builds x bitset_bits x top_k. The NestedLoops engine owns
# the run-grid product; targets run one configuration per invocation.
# ---------------------------------------------------------------------------
my @grid_dims = sort keys %{$run_matrix};
my $grid_iter = NestedLoops( [ map { $run_matrix->{$_} } @grid_dims ] );

my $total_cells = 0;
my $t0          = [gettimeofday];

CELL:
while ( my @point = $grid_iter->() ) {
  my %cell;
  @cell{@grid_dims} = @point;
  $total_cells++;

  for my $build (@builds) {
    my $spec = $build_matrix->{$build};
    my $cmd  = expand_cmd(
      $spec->{cmd},
      size        => $cell{bitset_bits},
      top_k       => $cell{top_k},
      num_queries => $sys->{num_queries},
      num_refs    => $cell{num_refs},
      iterations  => $iterations,
      threads     => $threads,
      gpu_id      => $sys->{gpu_id},
      python      => $python,
    );
    if ( $spec->{use_numactl} && length $numactl ) {
      $cmd = "$numactl $cmd";
    }

    print "[$build | bits=$cell{bitset_bits} k=$cell{top_k}] $cmd\n";
    next if $cli{dry_run};

    my @argv = shellwords($cmd);
    my ( $stdout, $stderr );
    # IPC::Run::run returns FALSE (does not die) on non-zero exit; capture that.
    my $ok = eval { IPC::Run::run( \@argv, '>', \$stdout, '2>', \$stderr ) };
    $ok = 0 if $@;    # IPC exception
    if ( !$ok ) {
      warn "WARNING: command failed for $build "
         . "(bits=$cell{bitset_bits} k=$cell{top_k}): $stderr\n";
      next;
    }

    my @rows = parse_iterations( $spec->{kind}, $stdout );
    if ( !@rows ) {
      warn "WARNING: no per-iteration rows parsed for $build "
         . "(bits=$cell{bitset_bits} k=$cell{top_k})\n";
      next;
    }
    my $n_threads = $spec->{device} eq 'CPU' ? $threads : 0;
    for my $row (@rows) {
      my ( $iter, $timing ) = @{$row};
      print {$out_fh} join( ',',
        $spec->{backend}, $build, $spec->{device},
        $cell{bitset_bits}, $cell{top_k},
        $sys->{num_queries}, $cell{num_refs},
        $n_threads, $iter, $timing
      ), "\n";
    }
  }
}

close $out_fh if $out_fh;
my $elapsed = tv_interval($t0);
printf "Done. %d grid cell(s) x %d build(s) in %.1f s%s\n",
  $total_cells, scalar(@builds), $elapsed,
  $cli{dry_run} ? " (dry run)" : " -> $results_csv";
