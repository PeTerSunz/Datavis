course = {"	Fundamentals of Computing" :"Jaruwan",
          "Data Analysis Using Spreadsheet Program" :"Jaruwan",
          "Selected Topics in Computer Science" :"Sukanya",
          "Computer Control and Audit" :"Worasait",
          "Badminton for Health" :"Somphol",
          "English Speak  " : "Jim",
            "English Spak  " : "Jim"}
course["Data Visualization"] = "Chatchai"
summary = {}
count = 0
# count2 = 0
# count3 = 0
# count4 = 0
# count5 = 0


for cname, name in course.items():
    print(f"Course : {cname} Instructor : {name} ")
    if name in summary:
        summary[name] = summary[name] + 1
    else:
        summary[name] = 1
    # if name == namesearch:
    #         count += 1
    # elif name == namesearch2:
    #         count2 += 1
    # elif name == namesearch3:
    #         count3 += 1
    # elif name == namesearch4:
    #         count4 += 1
    # elif name == namesearch5:
    #         count5 += 1
print("_________________________________")

for name, count in summary.items():
    print("%s Instructor  %d subjects" % (name, count))
# print("%s Instructor  %d subjects" %(namesearch2,count2))
# print("%s Instructor  %d subjects" %(namesearch3,count3))
# print("%s Instructor  %d subjects" %(namesearch4,count4))
# print("%s Instructor  %d subjects" %(namesearch5,count5))