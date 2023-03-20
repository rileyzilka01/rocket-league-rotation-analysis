import requests
import json

API_KEY = 't5E6suPaU3DWzepC78jNVOrYERrJclOKRPLrDrA7'

#names are gathered from the following link - higher earnings means overall better performance during the year
#https://www.esportsearnings.com/history/2022/games/409-rocket-league

NAMES = ["Firstkiller"]
PLAYLIST = 'ranked-duels'

def gather_replays():
    '''
    This method is for gathering replay id information from a subset of players defined by names
    '''
    
    headers = {'Authorization': API_KEY}

    for name in NAMES:
        print(f'STARTED getting replays for player {name}...')

        #get the reponse
        response = requests.get(f'https://ballchasing.com/api/replays?player-name={name}&minimum-rank=grand-champion-3&playlist={PLAYLIST}', headers=headers)

        #Get the replay id, name and link from the response
        i = 0
        replayIDs = {}
        replays = (response.json())['list']
        for replay in replays:
            replayIDs[i] = {'id': replay['id'], 'name': replay['replay_title'], "link": replay['link']}
            i += 1

        #write the replayIDs dict to a json file
        json_object = json.dumps(replayIDs, indent=4)

        with open(f'../data/replay-ids/{PLAYLIST}/replays-{name}.json', 'w') as f:
            f.write(json_object)

        print(f'DONE getting replays for {name}')

def download_positional_data():

    headers = {'Authorization': API_KEY}
    
    for name in NAMES:
        print(f'STARTED downloading positional data for player {name}...')

        f = open(f'../data/replay-ids/{PLAYLIST}/replays-{name}.json')
        data = json.load(f)

        for i in data:
            print(f'STARTED replay {i}...')

            fileID = data[i]['id']
            response = requests.get(f'https://ballchasing.com/dyn/replay/{fileID}/threejs', headers=headers)

            res = str(response.content)
            length = len(res)
            parsed = res[6:length-6]
            parsed = '{"' + parsed[0:10] + '":' + parsed[11:] + '}'

            parsed = json.loads(parsed)

            json_object = json.dumps(parsed, indent=4)

            with open(f'../data/positional-data/{PLAYLIST}/{name}-{i}.json', 'w') as f:
                f.write(json_object)

        print(f'FINISHED downloading positional data for player {name}')

def download_timeline_data():
    
    headers = {'Authorization': API_KEY}
    
    for name in NAMES:
        print(f'STARTED downloading positional data for player {name}...')

        f = open(f'../data/replay-ids/{PLAYLIST}/replays-{name}.json')
        data = json.load(f)

        for i in data:
            print(f"STARTED replay {i}...")

            fileID = data[i]['id']
            response = requests.get(f'https://ballchasing.com/dyn/replay/{fileID}/timeline', headers=headers)

            res = str(response.content)
            length = len(res)
            parsed = res[2:length-3]

            parsed = json.loads(parsed)

            json_object = json.dumps(parsed, indent=4)

            with open(f'../data/timeline-data/{PLAYLIST}/{name}-{i}.json', 'w') as f:
                f.write(json_object)

        print(f'FINISHED downloading positional data for player {name}')

def download_replay_stats():

    headers = {'Authorization': API_KEY}
    
    for name in NAMES:
        print(f'STARTED downloading replay stats for player {name}...')

        f = open(f'../data/replay-ids/{PLAYLIST}/replays-{name}.json')
        data = json.load(f)

        for i in data:
            print(f"STARTED replay {i}...")

            fileID = data[i]['id']
            response = requests.get(f'https://ballchasing.com/api/replays/{fileID}', headers=headers)

            response = response.json()

            json_object = json.dumps(response, indent=4)

            with open(f'../data/replay-stats/{PLAYLIST}/{name}-{i}.json', 'w') as f:
                f.write(json_object)

        print(f'FINISHED downloading replay stats for player {name}')

def download_replay_files():

    headers = {'Authorization': API_KEY}
    
    for name in NAMES:
        print(f'STARTED downloading replay files for player {name}...')

        f = open(f'../data/replay-ids/{PLAYLIST}/replays-{name}.json')
        data = json.load(f)

        for i in data:
            print(f"STARTED replay {i}...")

            fileID = data[i]['id']
            response = requests.get(f'https://ballchasing.com/api/replays/{fileID}/file', headers=headers)

            with open(f'../data/replays/{PLAYLIST}/{fileID}.replay', 'wb') as f:
                f.write(response.content)

        print(f'FINISHED downloading replay files for player {name}...')

def main():

    ans = input('(0) Stop program\n'
                '(1) Gather replay ids\n'
                '(2) Download positional data\n'
                '(3) Download timeline data\n'
                '(4) Download replay stats\n'
                '(5) Download replay files\n'
                'Choose an option: ')
    while ans != "0":
        if ans == "1":
            gather_replays()

        elif ans == "2":
            download_positional_data()

        elif ans == "3":
            download_timeline_data()

        elif ans == "4":
            download_replay_stats()

        elif ans == "5":
            download_replay_files()

        ans = input('(0) Stop program\n'
            '(1) Gather replay ids\n'
            '(2) Download positional data\n'
            '(3) Download timeline data\n'
            '(4) Download replay stats\n'
            '(5) Download replay files\n'
            'Choose an option: ')

    print('Terminating...')

if __name__ == "__main__":
    main()