
import json


NAMES = ['Firstkiller']
PLAYLIST = 'ranked-duels'
#COUNT is the highest number of replay in positional-data
COUNT = 40
PLAYERS = 2


def parse_balls(balls, compiled):

    for ball in balls:
        times = ball['times']
        positions = ball['pos']
        for i in range(len(times)):
            compiled[times[i]] = {'ball_x': positions[i*3],
                                  'ball_y': positions[(i*3)+1],
                                  'ball_z': positions[(i*3)+2]}

    return compiled

def parse_players(players, compiled):

    for i in range(len(players)):
        cars = players[i]['cars']
        for j in range(len(cars)):
            times = cars[j]['times']
            positions = cars[j]['pos']
            for k in range(len(times)):
                if times[k] in compiled:
                    compiled[times[k]] = compiled[times[k]] | {f'player_{i}_x': positions[k*3],
                                                               f'player_{i}_y': positions[(k*3)+1],
                                                               f'player_{i}_z': positions[(k*3)+2]}

    return compiled

def clean_data(compiled):

    remove = {}
    for time in compiled:
        for i in range(PLAYERS):
            if f'player_{i}_x' not in compiled[time]:
                if time not in remove:
                    remove[time] = ''

    for time in remove:
        compiled.pop(time)

    return compiled

def compile_positional_data():

    for name in NAMES:

        print(f'STARTING extraction for player {name}')

        for i in range(COUNT+1):

            print(f'STARTING file {i}...')

            compiled = {}

            f = open(f'../data/positional-data/{PLAYLIST}/{name}-{i}.json')
            data = json.load(f)

            compiled = parse_balls(data['replayData']['balls'], compiled)

            compiled = parse_players(data['replayData']['players'], compiled)

            compiled = clean_data(compiled)

            json_object = json.dumps(compiled, indent=4)

            with open(f'../data/extracted-positional-data/{PLAYLIST}/{i}.json', 'w') as f:
                f.write(json_object)

        print(f'FINISHED extraction for player {name}')

def main():
    compile_positional_data()

if __name__ == "__main__":
    main()