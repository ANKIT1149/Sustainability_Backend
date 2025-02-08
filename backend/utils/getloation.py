import requests


def get_address(lat, lon):
    nominatim_url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1}
    response = requests.get(
        nominatim_url, params=params, headers={"User-Agent": "my-recycling-app"}
    )
    if response.status_code == 200:
        data = response.json()
        return data.get("display_name", "Address not found")
    return "Error fetching address"


def find_nearest_recycling_center(lat, lon):
    overpass_url = "http://overpass-api.de/api/interpreter"

    query = f"""
     [out: json];
     node
       ["amenity"="recycling"]
       (around:10000, {lat}, {lon});
     out;
      """

    response = requests.get(overpass_url, params={"data": query})

    if response.status_code == 200:
        data = response.json()
        if "elements" in data and len(data["elements"]) > 0:
            centers = []
            for element in data["elements"]:
                center_lat = element["lat"]
                center_lon = element["lon"]

                center = {
                    "lat": center_lat,
                    "lon": center_lon,
                    "name": element.get("tags", {}).get(
                        "name", "Unnamed Recycling Center"
                    ),
                    "type": element.get("tags", {}).get("type", "General"),
                    "address": get_address(center_lat, center_lon),
                }
                centers.append(center)
            return centers
        else:
            return "No recycling centr found in this area"
    else:
        return f"Error {response.status_code}: {response.text}"


print(find_nearest_recycling_center(28.6139, 772090))
