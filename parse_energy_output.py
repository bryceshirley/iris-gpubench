import requests


def get_carbon_intensity():
    # URL of the API endpoint
    url = "https://api.carbonintensity.org.uk/regional"

    # Send GET request to the API
    response = requests.get(url, headers={'Accept': 'application/json'})

    # Parse the JSON response
    data = response.json()

    return data

# Function to get forecast and index for a given shortname
def get_forecast_and_index(data, shortname):
    regions = data['data'][0]['regions']
    for region in regions:
        if region['shortname'] == shortname:
            intensity = region['intensity']
            return intensity['forecast'], intensity['index']
    return None, None

# Function to print all shortname options
def print_shortname_options(data):
    regions = data['data'][0]['regions']
    print("Where are you running this benchmark?")
    for region in regions:
        print(region['shortname'])

# Example usage
data=get_carbon_intensity()
print_shortname_options(data)
print("\n")

shortname = input("Enter a shortname from the above options: ")
forecast, index = get_forecast_and_index(data, shortname)

if forecast is not None and index is not None:
    print(f"Forecast for {shortname}: {forecast}")
    print(f"Index for {shortname}: {index}")
else:
    print(f"Region with shortname '{shortname}' not found.")

