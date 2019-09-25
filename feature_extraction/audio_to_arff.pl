#!/usr/bin/env perl 
#=========================================================================
#  
#  This script file is used to extract opensmile features in batch
#  for wav files.
#
#  - Arguments: Configuration file which contains 
#  				1) directory path in the subfolders of which wav 
#  				   files are searched for recursively
#  				2) subfolders that need to search, multiple subfolders
#  				   are separated by white space. If no subfolders are
#  				   given, all subfolders are considered
#  				3) output file path
#  				For examples of configuration file, see iemocap.txt
#  				or msp.txt
# 
#  - Output: .arff file 
#
#=========================================================================

use strict;
use warnings;
use File::Basename;
use Term::ANSIColor qw(:constants);

# check if configuration file is given
my $arg_count = scalar(@ARGV);
if ($arg_count < 1) {
	die "Parameter missing";
}

# read configuration file
my $config_filename = $ARGV[0];
my $input_path = "";
my $output_path = "";
my @folders = [];
open(my $fh, '<:encoding(UTF-8)', $config_filename)
	or die "Could not open file '$config_filename'";
while (my $row = <$fh>) {
	if ($row =~ m/input_path/) {
		my @words = split /=/, $row;
		$input_path = $words[1];
	}
	if ($row =~ m/output_path/) {
		my @words = split /=/, $row;
		$output_path = $words[1];
	}
	if ($row =~ m/folders/) {
		my @words = split /=/, $row;
		@folders = split / /, $words[1];
	}
}
close $fh;
chomp $input_path;
chomp $output_path;

# set opensmile path and config file
my $home = "/mount/arbeitsdaten/asr-2/baofg";
my $openSMILE_root = "$home/opensmile-2.3.0";
my $SMILExtract = "$openSMILE_root/SMILExtract";
my $config = "$openSMILE_root/config/emobase2010.conf";

# overwrite the output file if exists, instead of appending
unlink($output_path);

# search for wav files recursively
traverse($input_path);

sub traverse {
	my ($path) = @_;
	return if not -d $path;
	opendir my $dh, $path or die "can't open $path";
	while (my $sub = readdir $dh) {
		next if $sub eq '.' or $sub eq '..';
		if (-d "$path/$sub") {
			$a = ($path ne $input_path) or (scalar(@folders) eq 0);
			$b = ($path eq $input_path and grep(/^$sub$/, @folders)); 
			if ($a or $b) {
				my @Wavs = glob("$path"."/".$sub."/*.wav");
				my $wav_count = scalar(@Wavs);
				print GREEN, "$path/$sub".": ".$wav_count." files\n", RESET;
				foreach my $wav (@Wavs) {
					my $filename = basename($wav);
					my $ret = system("$SMILExtract -C \"$config\"". 
					 		" -I \"$wav\" -O \"$output_path\"". 
							" -instname \"$filename\"");
					if ($ret eq 0) {
						print GREEN, "Successfully extracted features for ". 
						  "$filename\n", RESET;
					} else {
						print RED, "Error: $filename\n", RESET;
					}
				}
				traverse("$path/$sub");
			}
		}
	}
	close $dh;
	return;
}

if (-e $output_path) {
	print GREEN, "Output file: "."$output_path\n", RESET;
} else {
	print RED, "Failed to generate output file\n", RESET;
}
