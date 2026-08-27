#!/usr/bin/env perl
use v5.36; 
use File::Basename qw(dirname basename);
use File::Path     qw(make_path);
use File::Spec;
use Algorithm::Loops qw(NestedLoops);
use List::Util       qw(shuffle);
use Text::ParseWords qw(shellwords);
use Getopt::Long     qw(GetOptionsFromArray GetOptions);
use POSIX            qw(strftime);
use Sys::Hostname    qw(hostname);
use FindBin;
use IPC::Run         qw(run);
use Log::Log4perl    qw(:easy);
use JSON::PP         qw(decode_json);

# 1. Intercept Config File
my $config_file;
Getopt::Long::Configure("pass_through");
GetOptions('config=s' => \$config_file);
Getopt::Long::Configure("no_pass_through");
die "FATAL: --config <file.json> is strictly required.\n" unless $config_file;

# 2. Load JSON Schema
my %config;
eval {
    open my $fh, '<', $config_file or die "Cannot open: $!\n";
    my $json_text = do { local $/; <$fh> };
    close $fh;
    %config = %{ decode_json($json_text) };
};
die "FATAL: Failed to parse JSON config '$config_file':\n$@\n" if $@;

# 3. Dynamically Generate CLI Option Bindings
my %cli_opts;
my @getopt_spec;
for my $matrix (qw(build_matrix run_matrix system_env)) {
    next unless exists $config{$matrix};
    for my $key (keys %{ $config{$matrix} }) { push @getopt_spec, "$key=s"; }
}

GetOptions(\%cli_opts, @getopt_spec) or die "Error parsing CLI arguments.\n";
for my $k (keys %cli_opts) {
    next unless defined $cli_opts{$k};
    if (exists $config{build_matrix}{$k}) { $config{build_matrix}{$k} = $cli_opts{$k}; } 
    elsif (exists $config{run_matrix}{$k}) { $config{run_matrix}{$k} = $cli_opts{$k}; } 
    elsif (exists $config{system_env}{$k}) { $config{system_env}{$k} = $cli_opts{$k}; }
}

sub normalize_array {
    my ($val) = @_;
    return [] unless defined $val;
    return ref($val) eq 'ARRAY' ? $val : [ split(/\s*,\s*/, $val) ];
}
for my $k (keys %{$config{build_matrix}}) { $config{build_matrix}{$k} = normalize_array($config{build_matrix}{$k}); }
for my $k (keys %{$config{run_matrix}})   { $config{run_matrix}{$k}   = normalize_array($config{run_matrix}{$k}); }

# 4. Node & Instance Identification
my $hostname = hostname();
my $pid = $$;
my $mac_address = get_mac_address();
my $mac_clean = $mac_address;
$mac_clean =~ s/://g; 

my $machine_id = "Unknown";
if (open my $fh, '<', '/etc/machine-id') {
    $machine_id = <$fh>;
    chomp $machine_id;
    close $fh;
}

my $exec_context = $config{system_env}{exec_context};
if ($exec_context eq 'auto') {
    $exec_context = $ENV{SLURM_JOB_ID} // $ENV{PBS_JOBID} // $ENV{LSB_JOBID} // 'Manual';
}
my $run_id = strftime("%Y%m%d_%H%M%S", localtime) . "_$pid";

# 5. Execute Dynamic Telemetry
my %telemetry_data;
my @telemetry_keys;

for my $cat (keys %{ $config{telemetry} }) {
    my $node = $config{telemetry}{$cat};
    my $content = "";
    
    if ( $node->{file} && -e $node->{file} ) {
        open my $fh, '<', $node->{file} or next;
        $content = do { local $/; <$fh> };
        close $fh;
    } elsif ( $node->{cmd} ) {
        my $cmd = $node->{cmd};
        my $default_cc = $config{build_matrix}{cc}[0] // 'cc';
        $cmd =~ s/\{cc\}/$default_cc/g;
        eval { run [shellwords($cmd)], '>', \$content, '2>/dev/null'; };
    }
    
    for my $key (sort keys %{ $node->{extractors} }) {
        push @telemetry_keys, $key;
        my $regex = qr/$node->{extractors}{$key}/;
        $telemetry_data{$key} = ($content =~ $regex) ? $1 : "None";
    }
}

# 6. File Naming & Templating
my $proc_clean = $telemetry_data{Processor} // "UnknownCPU";
$proc_clean =~ s/\W+/_/g;
my $base_name = "${proc_clean}_${hostname}_${mac_clean}_${run_id}";

for my $k (keys %{$config{system_env}}) {
    if (defined $config{system_env}{$k} && !ref($config{system_env}{$k})) {
        $config{system_env}{$k} =~ s/\{base_name\}/$base_name/g;
        $config{system_env}{$k} =~ s/\{out_dir\}/$config{system_env}{out_dir}/g;
    }
}

# 7. Initialize Logging & I/O
make_path(dirname($config{system_env}{summary_csv}));
my $log_format = '[%d{yyyy-MM-dd HH:mm:ss}] [%p] %m%n';
Log::Log4perl->easy_init(
    { level => $DEBUG, file => "stdout", layout => $log_format },
    { level => $DEBUG, file => ">>$config{system_env}{raw_log}", layout => $log_format }
);

INFO("Starting Universal Benchmark Engine");
INFO("Run ID: $run_id | Node: $hostname ($mac_address) | Context: $exec_context");

my @b_keys = sort keys %{$config{build_matrix}};
my @r_keys = sort keys %{$config{run_matrix}};
my @cap_cols = @{ $config{system_env}{output_parser}{columns} };

unless (-e $config{system_env}{summary_csv}) {
    open my $fh, '>', $config{system_env}{summary_csv} or die "Cannot create CSV: $!\n";
    say $fh join(',', "Run_ID", "Machine_ID", "Hostname", "MAC_Address", "Exec_Context", 
                  @telemetry_keys, @b_keys, @r_keys, @cap_cols);
    close $fh;
}

# 8. Matrix Generation
my @build_arrays = map { $config{build_matrix}{$_} } @b_keys;
my @build_grid;
NestedLoops( \@build_arrays, sub { my %c; @c{@b_keys} = @_; push @build_grid, \%c; } );
@build_grid = shuffle(@build_grid);

my @run_arrays = map { $config{run_matrix}{$_} } @r_keys;
my @run_grid;
NestedLoops( \@run_arrays, sub { my %c; @c{@r_keys} = @_; push @run_grid, \%c; } );

# --- Execution Engine ---
for my $b_config (@build_grid) {
    next if $config{system_env}{dry_run};
    
    my $full_ctx = { %$b_config, %{$config{system_env}} };
    unless ( compile_binary($full_ctx) ) {
        WARN("Build failed for configuration. Skipping inner run matrix.");
        next;
    }

    for my $r_config (@run_grid) {
        $full_ctx = { %$b_config, %$r_config, %{$config{system_env}} };
        run_benchmark_instance($full_ctx, \@b_keys, \@r_keys);
    }
}

INFO("Sweep complete. Results stored at: $config{system_env}{summary_csv}");

# --- Core Subroutines ---

sub interpolate_cmd {
    my ($template, $ctx) = @_;
    $template =~ s/\{([^}]+)\}/defined $ctx->{$1} ? $ctx->{$1} : "{$1}"/ge;
    return $template;
}

sub compile_binary {
    my ($ctx) = @_;
    return 1 unless $ctx->{build_cmd};
    
    my $cmd_str = interpolate_cmd($ctx->{build_cmd}, $ctx);
    my @cmd = shellwords($cmd_str);
    
    INFO("BUILDING: " . join(' ', @cmd));
    my $build_log = '';
    eval { run \@cmd, '>', \$build_log, '2>&1' or die "Build returned non-zero status."; };
    if ($@) { ERROR("Compilation Exception:\n$@\nOutput:\n$build_log"); return 0; }
    return 1;
}

sub run_benchmark_instance {
    my ($ctx, $b_keys_ref, $r_keys_ref) = @_;

    if ( $ctx->{pre_run_cmd} ) {
        my $pre_cmd = interpolate_cmd($ctx->{pre_run_cmd}, $ctx);
        eval { run [shellwords($pre_cmd)], '>', \my $out, '2>&1'; };
    }

    my $run_str = interpolate_cmd($ctx->{run_cmd}, $ctx);
    my @run_args = shellwords($run_str);

    if ( $ctx->{priority} eq 'nice' ) { unshift @run_args, 'nice', '-n', '-20'; } 
    elsif ( $ctx->{priority} eq 'rr' ) { unshift @run_args, 'chrt', '-r', '50'; }
    if ( $ctx->{taskset} ) { unshift @run_args, 'taskset', '-c', $ctx->{taskset}; }

    DEBUG("RUNNING: " . join(' ', @run_args));

    my $exec_out = '';
    eval { run \@run_args, '>', \$exec_out, '2>&1' or die "Binary execution failed."; };
    if ($@) { WARN("Runtime Exception:\n$@\nOutput:\n$exec_out"); return; }

    my $parser = $ctx->{output_parser};
    my $compiled_regex = qr/$parser->{regex}/;
    
    open my $out, '>>', $ctx->{summary_csv} or die "Cannot append CSV: $!\n";
    for my $line ( split /\n/, $exec_out ) {
        if ( $line =~ $compiled_regex ) {
            my %caps = %+;
            
            # Apply dynamic mappings if they exist in the JSON
            if (exists $parser->{map}) {
                for my $col (keys %{ $parser->{map} }) {
                    my $val = defined $caps{$col} ? $caps{$col} : "__UNDEF__";
                    if (exists $parser->{map}{$col}{$val}) {
                        $caps{$col} = $parser->{map}{$col}{$val};
                    }
                }
            }
            
            my @t_vals = map { $telemetry_data{$_} } @telemetry_keys;
            my @b_vals = map { $ctx->{$_} } @$b_keys_ref;
            my @r_vals = map { $ctx->{$_} } @$r_keys_ref;
            my @cap_vals = map { $caps{$_} // '' } @{ $parser->{columns} };
            
            say {$out} join(',', $run_id, $machine_id, $hostname, $mac_address, $exec_context, 
                                 @t_vals, @b_vals, @r_vals, @cap_vals);
        }
    }
    close $out;
}

sub get_mac_address {
    my $preferred;   # interface with default route (best) #[cite: 4]
    my $permanent;   # permanent physical Ethernet #[cite: 4]
    my $fallback;    # any reasonable Ethernet MAC #[cite: 4]

    if (open my $rt, '<', '/proc/net/route') { #[cite: 4]
        while (<$rt>) { #[cite: 4]
            my ($iface, $dest) = (split)[0,1]; #[cite: 4]
            next unless defined $iface && $dest eq '00000000'; #[cite: 4]
            $preferred = $iface; #[cite: 4]
            last; #[cite: 4]
        }
        close $rt; #[cite: 4]
    }

    opendir my $dh, '/sys/class/net' or return "00:00:00:00:00:00"; #[cite: 4]
    for my $iface (readdir $dh) { #[cite: 4]
        next if $iface eq '.' || $iface eq '..' || $iface eq 'lo'; #[cite: 4]
        my $base = "/sys/class/net/$iface"; #[cite: 4]
        
        next unless -e "$base/device" || -e "$base/phy80211"; #[cite: 4]

        my $type = do { #[cite: 4]
            open my $t, '<', "$base/type" or next; #[cite: 4]
            local $/; <$t> + 0; #[cite: 4]
        };
        next unless $type == 1; #[cite: 4]

        open my $af, '<', "$base/address" or next; #[cite: 4]
        my $mac = <$af>; #[cite: 4]
        close $af; #[cite: 4]
        chomp $mac; #[cite: 4]
        $mac = lc $mac; #[cite: 4]
        next if $mac eq '00:00:00:00:00:00' || $mac !~ /^([0-9a-f]{2}:){5}[0-9a-f]{2}$/; #[cite: 4]

        my $assign = do { #[cite: 4]
            open my $a, '<', "$base/addr_assign_type" or next; #[cite: 4]
            local $/; <$a> + 0; #[cite: 4]
        };

        if (defined $preferred && $iface eq $preferred) { #[cite: 4]
            closedir $dh; #[cite: 4]
            return $mac; #[cite: 4]
        }
        $permanent = $mac if $assign == 0 && !defined $permanent; #[cite: 4]
        $fallback  = $mac unless defined $fallback; #[cite: 4]
    }
    closedir $dh; #[cite: 4]

    return $permanent // $fallback // "00:00:00:00:00:00"; #[cite: 4]
}