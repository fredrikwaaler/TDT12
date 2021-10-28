def slugify(value):
    """
    Makes a string value valid for filename
    """
    new_value = str(value)
    invalid = '<>"!\|/?* '

    for char in invalid:
        new_value = new_value.replace(char, '')

    return new_value
