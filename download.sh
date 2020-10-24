# Fetch data and put it in data folder
curl -fsS http://udon.stacken.kth.se/\~ninjin/comp0090_assignment_1_data.tar.gz -o ./data.tar.gz

# Extract data 
tar -xvf data.tar.gz

# Remove unneeded file
rm -f ./data.tar.g

# Rename folder
mv comp0090* data

