def dashboard(cur):
    try:
        print("hi")
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        # Fetch all the rows
        tables = cur.fetchall()
        
        # Print the table names
        for table in tables:
            print(table[0])
        
        return tables
    except Exception as e:
        print("Error fetching data:", e) 
        return []
