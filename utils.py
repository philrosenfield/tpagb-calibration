def check_astcor(filters):
    """add _cor to filter names if it isn't already there"""
    if type(filters) is str:
        filters = [filters]

    for i, f in enumerate(filters):
        if not f.endswith('cor'):
            filters[i] = f + '_cor'
    return filters