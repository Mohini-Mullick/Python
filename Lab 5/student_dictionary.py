# Create a student dictionary
student = {
    "first_name": "Mohini",
    "last_name": "Mullick",
    "gender": "Female",
    "age": 21,
    "marital_status": "Single",
    "skills": ["Python", "Data Analysis"],
    "country": "India",
    "city": "Kol",
    "address": "kolorah"
}
length_of_dict = len(student)
print("Length of the student dictionary:", length_of_dict)
skills = student["skills"]
print("Skills:", skills)
print("Data type of skills:", type(skills))
student["skills"].extend(["Machine Learning", "Web Development"])
print("Modified skills:", student["skills"])
keys_list = list(student.keys())
print("Keys in the dictionary:", keys_list)
values_list = list(student.values())
print("Values in the dictionary:", values_list)
tuples_list = list(student.items())
print("Dictionary as a list of tuples:", tuples_list)
del student["marital_status"]
print("Dictionary after deleting marital_status:", student)
del student

