def append_multiple_values(base_list, *single_values, *value_sequences):
    # Append single values
    for value in single_values:
        base_list.append(value)
    
    # Extend with sequences
    for sequence in value_sequences:
        base_list.extend(sequence)
    
    return base_list
