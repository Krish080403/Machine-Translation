f = open("input.txt")
o = open("input_processed.txt","w")
for line in f:
	splitline = line.split("। ")
	for s in splitline:
		o.write(s+"\n")
o.close()
