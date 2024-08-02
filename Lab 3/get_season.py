def get_season(month):
    month = month.lower()
    
    if month in ['december', 'january', 'febuary']:
        return 'Winter'
    if month in ['march', 'april', 'may']:
        return 'Spring'
    if month in ['june', 'july', 'august']:
        return 'Summer'
    if month in ['september', 'october', 'november']:
        return 'Autumn'
    
    return 'Invalid month!'

print(get_season('December'))

print(get_season('March'))

print(get_season('October'))

print(get_season('anonexistent month'))