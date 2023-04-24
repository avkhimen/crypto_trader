# Preprocessing data

# Extract the values of the cur1, cur2, and price_type fields from config.yml
cur1=$(grep -E '^cur1:' config.yml | awk '{print $2}')
cur2=$(grep -E '^cur2:' config.yml | awk '{print $2}')
price_type=$(grep -E '^price_type:' config.yml | awk '{print $2}')

python3 data_preprocessing.py -c1 $cur1 -c2 $cur2 -p $price_type

# Getting the name of the expected processed file
filename="$cur1"_"$cur2"_"$price_type".csv
