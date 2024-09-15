from rag_brain import chroma_db
from preprocess import FA_data


print("db = load_db_main()")

def load_db_main():
    db = chroma_db.Store()
    if db.count == 0:
        print("Found empty db, getting data...")
        input("Hit enter for okay")
        data = FA_data.get()
        print("Retrieved data, adding it...")
        db.read(data)
        db.count = db.collection.count()
        print(f"Added {db.count} items.")
    else:
        print(f"Found old database with {db.count} items.")
        input("Hit enter for okay")
    return db




