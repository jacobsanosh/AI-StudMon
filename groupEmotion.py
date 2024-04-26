import db_connect
import collections

connection = db_connect.connect_to_supabase()
cur = connection[0]
conn = connection[1]

def processingEmotion(tname):
    try:
        student = collections.defaultdict(list)

        # Execute SQL query to fetch data from the database
        query = f"SELECT student_name, emotion FROM {tname}"
        cur.execute(query)

        # Fetch all rows from the result set
        rows = cur.fetchall()

        # Process the fetched data
        for row in rows:
            # Check if the emotion is "Happy" or "Neutral"
            if row[1] in ["Happy", "Neutral"]:
                student[row[0]].append(1)
            else:
                student[row[0]].append(-1)
        
        print(student)
        majority_emotion = {}
        for name, emotions in student.items():
            count_1 = emotions.count(1)
            count_minus_1 = emotions.count(-1)
            if count_1 > count_minus_1:
                majority_emotion[name] = 1
            elif count_minus_1 > count_1:
                majority_emotion[name] = -1
            else:
                majority_emotion[name] = 1
        
        print(majority_emotion)
        count_1,count_minus_1=0,0
        for name, emotion in majority_emotion.items():
            if emotion == 1:
                count_1+=1
            else:
                count_minus_1+=1
        res=0
        if count_1 > count_minus_1:
            res= 1
        elif count_minus_1 > count_1:
            res= -1
        else:
            res= 1
        res="understood" if res==1 else "not understood"
        print(res)
    except Exception as e:
        print("Error fetching data:", e)

processingEmotion("class_20240426_091949")
