import requests


# we want to exclude deputies from unknown/noname clubs
CLUBS_WHITELIST = ["Razem", "Lewica", "Polska2050-TD", "PSL-TD", "KO", "PiS", "Konfederacja"]

DEPUTIES_URI = "https://api.sejm.gov.pl/sejm/term10/MP"

response = requests.get(DEPUTIES_URI)
response.raise_for_status()
deputies = response.json()

members_by_club = {club: [] for club in CLUBS_WHITELIST}
for deputy in deputies:
    club = deputy.get("club")
    firstLastName = deputy.get("firstLastName")
    id = deputy.get("id")
    if club in CLUBS_WHITELIST and firstLastName and id:
        info = {"firstLastName": firstLastName, "id": id}
        members_by_club.setdefault(club, []).append(info)
    
print(members_by_club)

# https://api.sejm.gov.pl/sejm/term10/proceedings/1/2023-11-13/transcripts