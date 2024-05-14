
import matplotlib.pyplot as plt
import io

def derive_graph(table_name, cur):
    # Create a query to fetch data from the Supabase table with specific columns
    query = f"SELECT student_name, emotion, timestamp FROM {table_name}"
    cur.execute(query)
    data = cur.fetchall()

    # Your analytics logic here
    emotions_order = {
        "Happy": 3,
        "Neutral": 2,
        "Sad": 1,
        "Fear": -1,
        "Surprise": -2,
        "Angry": 0,
    }

    # Group data by student_name
    grouped_data = {}
    if len(data) == 0:
        print("No data found")
        return []

    for student, emotion, timestamp in data:
        student_name = student
        if student_name.lower() != "unknown":
            if student_name not in grouped_data:
                grouped_data[student_name] = {"x": [], "y": []}
            grouped_data[student_name]["x"].append(timestamp)
            grouped_data[student_name]["y"].append(emotion)

    # Plot creation for each student
    plot_bytes_list = []
    for student_name, values in grouped_data.items():
        plt.figure()  # Create a new plot for each student
        x_values = values["x"]
        y_values = values["y"]

        # Convert emotion labels to their corresponding numeric values for plotting
        y_numeric_values = [emotions_order.get(emotion, -1) for emotion in y_values]

        # Sort the data based on timestamps to ensure a chronological order
        sorted_data = sorted(zip(x_values, y_numeric_values))
        sorted_x_values, sorted_y_values = zip(*sorted_data)

        plt.plot(
            sorted_x_values,
            sorted_y_values,
            marker="o",
            linestyle="-",
        )

        plt.xlabel("Timestamp")
        plt.ylabel("Emotion")
        plt.title(f"Understanding Graph for {student_name} - {table_name}")
        plt.xticks(rotation=45)
        plt.yticks(
            list(emotions_order.values()), emotions_order.keys()
        )  # Set custom y-axis labels
        plt.tight_layout()

        # Save plot to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        plt.close()

        plot_bytes_list.append(img_buffer.getvalue())

    return plot_bytes_list
