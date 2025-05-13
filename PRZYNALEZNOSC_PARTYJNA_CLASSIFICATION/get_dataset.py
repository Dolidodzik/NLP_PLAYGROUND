import requests
from datetime import datetime
from bs4 import BeautifulSoup


CLUBS_WHITELIST = ["Razem", "Lewica", "Polska2050-TD", "PSL-TD", "KO", "PiS", "Konfederacja"] # we want to exclude deputies from unknown/noname clubs
OLDEST_DATE = "2025-01-01" # we only want speeches from 2025 or newer
NEWEST_DATE = "2025-05-12" # we dont want speeches newer than today (we exclude planned proceedings that havent yet take place), cannot be newer than yesterday


OLDEST_DATETIME = datetime.strptime(OLDEST_DATE, "%Y-%m-%d")
NEWEST_DATETIME = datetime.strptime(NEWEST_DATE, "%Y-%m-%d")


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
        info = {"firstLastName": firstLastName, "id": id, "statements": []}
        members_by_club.setdefault(club, []).append(info)
    
print("\nfound members: ")
print(members_by_club)


# getting list of proceedings from 2023-2027 cadence (10)
PROCEEDINGS_URL = "https://api.sejm.gov.pl/sejm/term10/proceedings/"

response = requests.get(PROCEEDINGS_URL)
response.raise_for_status()
raw_proceedings = response.json()

filtered_proceedings = []
for raw_proceeding in raw_proceedings:
    too_old = False
    for date in raw_proceeding['dates']:
        if datetime.strptime(date, "%Y-%m-%d") < OLDEST_DATETIME or datetime.strptime(date, "%Y-%m-%d") > NEWEST_DATETIME:
            too_old = True
    
    if not too_old:
        filtered_proceedings.append({
            "number": raw_proceeding['number'],
            "dates": raw_proceeding['dates'],
            "title": raw_proceeding['title']
        })

print("\nfound filtered proceedings:")
print(filtered_proceedings)

def extract_blockquote_text(html: str) -> str:
    '''
    parses HTML from transcripts texts like this one - https://api.sejm.gov.pl/sejm/term10/proceedings/7/2024-03-07/transcripts/3
    outputs clean text without html tags, just speech text
    '''
    soup = BeautifulSoup(html, 'html.parser')
    blk = soup.find('blockquote')
    if not blk:
        return ""
    
    for h1 in blk.find_all('h1'):
        h1.decompose()
    
    for el in blk.select('.przebieg'):
        el.decompose()

    for el in blk.select('.mowca'):
        el.decompose()
    
    for el in blk.select('.punkt-tytul'):
        el.decompose()
    
    for el in blk.select('.punkt'):
        el.decompose()
    
    text = blk.get_text(separator='\n', strip=True)
    return text

for filtered_proceeding in filtered_proceedings:
   for date in filtered_proceeding['dates']:
        print("getting transcripts lists for date: ",)
        TRANSCRIPTS_LIST_URL = f"https://api.sejm.gov.pl/sejm/term10/proceedings/{filtered_proceeding['number']}/{date}/transcripts/"
        response = requests.get(TRANSCRIPTS_LIST_URL)
        response.raise_for_status()
        raw_transcripts_list = response.json()

        for statement in raw_transcripts_list['statements']:
            TRANSCRIPTS_RAW_TEXT_URL = f"https://api.sejm.gov.pl/sejm/term10/proceedings/{filtered_proceeding['number']}/{date}/transcripts/{statement['num']}"
            response = requests.get(TRANSCRIPTS_RAW_TEXT_URL)
            response.raise_for_status()
            raw_transcript_text = response.text

            pure_text = extract_blockquote_text(raw_transcript_text)
            if pure_text:
                print("hoorayy we have good pure text")
            else:
                print("No <blockquote> found. Ignoring that statement")




# https://api.sejm.gov.pl/sejm/term10/proceedings/1/2023-11-13/transcripts