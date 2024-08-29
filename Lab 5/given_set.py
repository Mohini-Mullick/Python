# Initializing the sets and list
it_companies = {'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon'}
A = {19, 22, 24, 20, 25, 26}
B = {19, 22, 20, 25, 26, 24, 28, 27}
age = [22, 19, 24, 25, 26, 24, 25, 24]
length_it_companies = len(it_companies)
print(f"Length of it_companies set: {length_it_companies}")
it_companies.add('Twitter')
print(f"it_companies after adding 'Twitter': {it_companies}")
new_companies = {'LinkedIn', 'Snapchat', 'Pinterest'}
it_companies.update(new_companies)
print(f"it_companies after adding new companies: {it_companies}")
it_companies.remove('Oracle')  
print(f"it_companies after removing 'Oracle': {it_companies}")
try:
    it_companies.remove('NonExistentCompany')
except KeyError:
    print("Tried to remove a non-existent company with 'remove' and caught a KeyError")
it_companies.discard('NonExistentCompany')  
print(f"it_companies after attempting to discard a non-existent company: {it_companies}")
