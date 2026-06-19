#!/usr/bin/perl
use strict;
use warnings;
use Getopt::Long;
use IPC::Open3;
use Symbol qw(gensym);

my @gcc_versions;
my @clang_versions;

GetOptions(
    "gcc=s"   => sub { @gcc_versions = split(',', $_[1]) },
    "clang=s" => sub { @clang_versions = split(',', $_[1]) },
) or die "Usage: $0 --gcc 15,14 --clang 19\n";

my %accepted;
my %conflicts;

sub simulate_apt {
    my (@pkgs) = @_;
    my ($in, $out, $err) = (gensym, gensym, gensym);
    my $pid = open3($in, $out, $err, 'apt-get', '-s', 'install', @pkgs);
    close $in;
    
    my $output = do { local $/; <$out> };
    my $error  = do { local $/; <$err> };
    waitpid($pid, 0);
    
    # Extract specific conflict line if it exists
    my $conflict_msg = "Unknown dependency error.";
    if ($error =~ /(Conflicts:.*|Depends:.*|unmet dependencies.*)/i) {
        $conflict_msg = $1;
    }
    return ($output, $error, $conflict_msg);
}

sub process_queue {
    my ($type, $requested_ref) = @_;
    my @queue = @$requested_ref;
    my @running_list;

    foreach my $ver (@queue) {
        print "-> Simulating dependencies for $type-$ver... ";
        my @new_pkgs = ($type eq 'gcc') ? 
            ("gcc-$ver", "g++-$ver", "gcc-$ver-offload-nvptx", "gcc-$ver-offload-amdgcn") :
            ("clang-$ver", "libomp-$ver-dev");

        my @test_list = (@running_list, @new_pkgs);
        my ($out, $err, $msg) = simulate_apt(@test_list);

        if ($out =~ /unmet dependencies|Conflicts:|broken packages/i) {
            print "FAILED.\n";
            $conflicts{$type}{$ver} = $msg;
        } else {
            print "OK.\n";
            push @running_list, @new_pkgs;
            $accepted{$type} = [@running_list];
        }
    }
}

# --- Execution ---
if (@gcc_versions)   { process_queue('gcc',   \@gcc_versions);   }
if (@clang_versions) { process_queue('clang', \@clang_versions); }

# --- Reporting ---
print "\n" . "="x50 . "\n PRIORITY RESOLUTION REPORT \n" . "="x50 . "\n";
foreach my $type (keys %accepted) {
    print "\nACCEPTED $type PACKAGES:\n";
    print " - $_\n" for @{$accepted{$type}};
}

foreach my $type (keys %conflicts) {
    print "\nREJECTED $type VERSIONS:\n";
    foreach my $ver (keys %{$conflicts{$type}}) {
        print " - Version $ver: $conflicts{$type}{$ver}\n";
    }
}
print "\nProceed with installation? [y/N]: ";
my $answer = <STDIN>;
if ($answer =~ /^y/i) {
    my @final_install = (@{$accepted{gcc} // []}, @{$accepted{clang} // []});
    if (@final_install) {
        system('sudo', 'apt-get', 'install', '-y', @final_install);
        print "\nInstallation complete. Configure update-alternatives manually as needed.\n";
    }
} else {
    print "Aborted.\n";
}
