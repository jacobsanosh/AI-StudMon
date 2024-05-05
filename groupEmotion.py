# import db_connect
import collections
from datetime import datetime, timedelta
# connection = db_connect.connect_to_supabase()
# cur = connection[0]
# conn = connection[1]

def processingEmotion(cur,tname,end_time):
    try:
        student = collections.defaultdict(list)
        start_time = end_time - timedelta(minutes=10)
        print(start_time," to ",end_time)
        # Execute SQL query to fetch data from the database within the specified time interval
        query = f"SELECT student_name, emotion FROM {tname} WHERE timestamp BETWEEN %s AND %s AND student_name != 'Unknown'"
        # query = f"SELECT student_name, emotion FROM {tname} WHERE timestamp BETWEEN %s AND %s"
        cur.execute(query, (start_time, end_time))

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
        if not majority_emotion:
            print("No data found most of them are unknown students.")
        else: 
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
            return res
    except Exception as e:
        print("Error fetching data:", e)

#for checking


# end_time = datetime.strptime("2024-04-26 9:21:00", "%Y-%m-%d %H:%M:%S")
# print(end_time)
# processingEmotion("class_20240426_091949",end_time)
# end_time = datetime.strptime("2024-04-26 15:43:00", "%Y-%m-%d %H:%M:%S")
# print(end_time)
# processingEmotion("class_20240426_154218",end_time)
