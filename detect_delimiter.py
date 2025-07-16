import csv

# Mapping common delimiters to their names
delimiter_names = {
    ',': 'comma',
    '\t': 'tab',
    ' ': 'space',
    ';': 'semicolon',
    '|': 'pipe',
}

with open('Dataset/user_artists.dat', 'r') as f:
    sample = f.read(1024)
    dialect = csv.Sniffer().sniff(sample)
    delim_char = dialect.delimiter

    # Get the delimiter name or show 'unknown' if not in the map
    delim_name = delimiter_names.get(delim_char, f"unknown ('{delim_char}')")

    print(f"Detected delimiter: {delim_name}")
