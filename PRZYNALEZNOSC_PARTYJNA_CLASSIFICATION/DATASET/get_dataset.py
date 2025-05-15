import requests
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd


CLUBS_WHITELIST = ["Razem", "Lewica", "Polska2050-TD", "PSL-TD", "KO", "PiS", "Konfederacja"] # we want to exclude deputies from unknown/noname clubs
OLDEST_DATE = "2023-12-12"
NEWEST_DATE = "2025-05-12"


OLDEST_DATETIME = datetime.strptime(OLDEST_DATE, "%Y-%m-%d")
NEWEST_DATETIME = datetime.strptime(NEWEST_DATE, "%Y-%m-%d")


DEPUTIES_URI = "https://api.sejm.gov.pl/sejm/term10/MP"

response = requests.get(DEPUTIES_URI)
response.raise_for_status()
deputies = response.json()

# Create a mapping of member ID to club and name for later use
member_info = {}
for deputy in deputies:
    club = deputy.get("club")
    firstLastName = deputy.get("firstLastName")
    id = deputy.get("id")
    if club in CLUBS_WHITELIST and firstLastName and id:
        member_info[id] = {
            "firstLastName": firstLastName,
            "club": club
        }

# Create an empty DataFrame with the required columns
df = pd.DataFrame(columns=["member_id", "first_last_name", "club", "statement", "date"])
    
print("\nfound members: ")
print(member_info)


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
        print(f"getting transcripts lists for date: {date}")
        TRANSCRIPTS_LIST_URL = f"https://api.sejm.gov.pl/sejm/term10/proceedings/{filtered_proceeding['number']}/{date}/transcripts/"
        response = requests.get(TRANSCRIPTS_LIST_URL)
        response.raise_for_status()
        raw_transcripts_list = response.json()

        for statement in raw_transcripts_list['statements']:
            try:
                TRANSCRIPTS_RAW_TEXT_URL = f"https://api.sejm.gov.pl/sejm/term10/proceedings/{filtered_proceeding['number']}/{date}/transcripts/{statement['num']}"
                response = requests.get(TRANSCRIPTS_RAW_TEXT_URL)
                response.raise_for_status()
                raw_transcript_text = response.text

                pure_text = extract_blockquote_text(raw_transcript_text)
                member_id = statement.get('memberID')
                
                if pure_text and member_id in member_info:
                    # Add a new row to the DataFrame for each statement
                    new_row = {
                        "member_id": member_id,
                        "first_last_name": member_info[member_id]["firstLastName"],
                        "club": member_info[member_id]["club"],
                        "statement": pure_text,
                        "date": date
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            except Exception as e:
                print("cos sie zjebalo ", e) 
                continue

# Save the DataFrame to a CSV file
df.to_csv('FULL_DATASET_ORIGINAL_2024_TO_MAY_2025.csv', index=False, encoding='utf-8')

print(f"Total statements collected: {len(df)}")
print("Data saved to statements_data.csv")