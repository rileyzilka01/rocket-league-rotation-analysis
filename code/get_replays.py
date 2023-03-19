import requests
import json

API_KEY = 't5E6suPaU3DWzepC78jNVOrYERrJclOKRPLrDrA7'

#names are gathered from the following link - higher earnings means overall better performance during the year
#https://www.esportsearnings.com/history/2022/games/409-rocket-league

NAMES = ["Caard", "Chicago", "Jstn", "Firstkiller"]

def gather_replays():
    '''
    This method is for gathering replay id information from a subset of players defined by names
    '''
    
    headers = {'Authorization': API_KEY}

    for name in NAMES:
        print(f'STARTED getting replays for player {name}...')

        #get the reponse
        response = requests.get(f'https://ballchasing.com/api/replays?player-name={name}&minimum-rank=grand-champion-3', headers=headers)

        #Get the replay id, name and link from the response
        i = 0
        replayIDs = {}
        replays = (response.json())['list']
        for replay in replays:
            replayIDs[i] = {'id': replay['id'], 'name': replay['replay_title'], "link": replay['link']}
            i += 1

        #write the replayIDs dict to a json file
        json_object = json.dumps(replayIDs, indent=4)

        with open(f'../data/replays/replays-{name}.json', 'w') as f:
            f.write(json_object)

        print(f'DONE getting replays for {name}')

def download_replays():
    pass


def main():

    ans = input('(0) Stop program\n(1) Gather replay ids\n(2) Download replay files\nChoose an option: ')
    while ans != "0":
        if ans == "1":
            gather_replays()
        elif ans == "2":
            download_replays()
        ans = input('(0) Stop program\n(1) Gather replay ids\n(2) Download replay files\nChoose an option: ')

    print('Terminating...')

if __name__ == "__main__":
    main()