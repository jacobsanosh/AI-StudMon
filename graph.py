# analytics.py

from supabase_py import create_client
import matplotlib.pyplot as plt
import io


def derive_graph(table_name,cur):
    try:
        query = f"SELECT student_name, emotion,timestamp FROM {table_name}"
        cur.execute(query)
        emotions_order = {"Happy": 3, "Neutral": 2, "Sad": 1, "sad": 1, "Angry": 0}

        data = cur.fetchall()
        grouped_data = {}
        for student, emotion,timestamp in data:
            student_name= student
            if student_name not in grouped_data:
                    grouped_data[student_name] = {"x": [], "y": []}
                    grouped_data[student_name]["x"].append(timestamp)
                    grouped_data[student_name]["y"].append(emotion)



        # Fetch data from the Supabase table
        # response = supabase.table(table_name).select().execute()
        # data = response["data"]

        # Your analytics logic here

        # Group data by student_name
        # grouped_data = {}
        # for item in data:
        #     student_name = item["student_name"]
        #     if student_name not in grouped_data:
        #         grouped_data[student_name] = {"x": [], "y": []}
        #     grouped_data[student_name]["x"].append(item["timestamp"])
        #     grouped_data[student_name]["y"].append(item["emotion"])

        # Plot creation
        # Plot creation
        plt.figure()
        for student_name, values in grouped_data.items():
            # Reorder y-axis data based on custom sorting order
            x_values = values["x"]
            y_values = values["y"]
            reordered_y_values = sorted(
                y_values, key=lambda x: emotions_order.get(x, -1)
            )  # Sorting based on custom emotion order
            reordered_x_values = [
                x
                for _, x in sorted(
                    zip(y_values, x_values),
                    key=lambda pair: emotions_order.get(pair[0], -1),
                )
            ]
            plt.plot(
                reordered_x_values,
                reordered_y_values,
                marker="o",
                linestyle="-",
                label=student_name,
            )

        plt.xlabel("Timestamp")
        plt.ylabel("Emotion")
        plt.title(f"Understanding Graph for {table_name}")
        plt.xticks(rotation=45)
        plt.yticks(
            range(len(emotions_order)), emotions_order.keys()
        )  # Set custom y-axis labels
        plt.legend()
        plt.tight_layout()

        # Save plot to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        plt.close()

        return img_buffer.getvalue()
    
    except Exception as e:
        print("Errorr fetching data:", e)

