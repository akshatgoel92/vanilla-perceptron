# Go to code repository and make data folder
cd intro-dl && mkdir data

# Install requirements
pip install -r requirements.txt

# Fetch data and put it in data folder
curl -fsS http://udon.stacken.kth.se/\~ninjin/comp0090_assignment_1_data.tar.gz -o /data/data.tar.gz

# Extract data 
tar -x -z -f /data/data.tar.gz

# Remove unneeded file
rm -f /data/data.tar.gz