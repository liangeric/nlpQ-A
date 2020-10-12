#! /usr/bin/perl
#

open(F, "topics.txt");

$i = 1;
while (<F>)
{
	chomp;
	$line = $_;

	system("../../support/getwikitext.pl $line 1 1 > a$i.txt.clean");
	system("../../support/getwikitext.pl $line 1 > a$i.txt");
	system("../../support/getwikitext.pl $line > a$i.htm");
	system("../../support/getwikitext.pl $line 0 1 >  a$i"."o.htm");

	$i++;
}

close(F);
