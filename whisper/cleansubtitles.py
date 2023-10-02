import re
import sys

# Get filename from command line argument
filename = sys.argv[1] 

# Read in file
with open(filename, 'r') as f:
	text = f.read()

# Extract timestamps and text    
entries = re.findall(r'(\d+)\n(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)\n(.*)', text)
preventry=("", "", "" "")
counter=0
newentries=[]
lasttimestamp=""
counter=0
for data in entries:
	if counter==0:
		preventry=data
		counter+=1
		starttime=data[1]
		endtime=data[2]
	else:
		if data[3]==preventry[3]:
			endtime=data[2]
		else:
			newentries.append((str(counter), starttime, endtime, preventry[3]))
			starttime=data[1]
			endtime=data[2]
			preventry=data
			counter+=1
		

	'''if not data[3]==preventry:
		millisecond=int(data[2].split(",")[1])
		time=data[2].split(",")[0]
		hour=int(time.split(":")[0])
		minute=int(time.split(":")[1])
		second=int(time.split(":")[2])
		
		currenttime=3600*hour+60*minute+second+millisecond/1000
		
		newentries.append((counter, data[1], lasttimestamp, data[3]))
		counter+=1

	preventry=data[3]
	lasttimestamp=data[2]'''

for data in newentries:
	print(data[0]+"\n"+ data[1] + " --> " + data[2] + "\n" + data[3]+"\n")
	
