# Define the person dictionary
person = {
    'first_name': 'Asabeneh',
    'last_name': 'Yetayeh',
    'age': 250,
    'country': 'Finland',
    'is_married': True,
    'skills': ['JavaScript', 'React', 'Node', 'MongoDB', 'Python'],
    'address': {
        'street': 'Space street',
        'zipcode': '02210'
    }
}
if 'skills' in person:
    skills = person['skills']
    middle_index = len(skills) // 2
    print('Middle skill:', skills[middle_index])
if 'skills' in person:
    has_python = 'Python' in person['skills']
    print('Has Python skill:', has_python)
if 'skills' in person:
    skills_set = set(person['skills'])
    if skills_set == {'JavaScript', 'React'}:
        print('He is a front end developer')
    elif skills_set >= {'Node', 'Python', 'MongoDB'}:
        print('He is a backend developer')
    elif skills_set >= {'React', 'Node', 'MongoDB'}:
        print('He is a fullstack developer')
    else:
        print('unknown title')
if person.get('is_married') and person.get('country') == 'Finland':
    print(f"{person['first_name']} {person['last_name']} lives in Finland. He is married.")
