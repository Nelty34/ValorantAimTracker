import PySimpleGUI as sg
import json
import os
from pathlib import Path
from ValorantAimTracker import detect_enemies_in_video
import multiprocessing as mp

# Set theme
sg.theme('DarkBlue3')

# File paths
USERS_FILE = 'users.json'
VIDEO_FOLDER = os.path.dirname(__file__)

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_user_window():
    """Create a window for new user creation"""
    layout = [
        [sg.Text('Create New User', font=('Arial', 14, 'bold'))],
        [sg.Text('Player Name:'), sg.InputText(key='PLAYER_NAME', size=(20, 1))],
        [sg.Text('Player Tag (e.g., #1234):'), sg.InputText(key='PLAYER_TAG', size=(20, 1))],
        [sg.Button('Create'), sg.Button('Cancel')]
    ]
    
    window = sg.Window('Create New User', layout)
    
    while True:
        event, values = window.read()
        
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            window.close()
            return None
        
        if event == 'Create':
            if values['PLAYER_NAME'].strip() and values['PLAYER_TAG'].strip():
                window.close()
                return {
                    'name': values['PLAYER_NAME'],
                    'tag': values['PLAYER_TAG'],
                    'matches': []
                }
            else:
                sg.popup_error('Please fill in all fields')

def create_main_window():
    """Create the main application window"""
    users = load_users()
    user_list = list(users.keys()) if users else []
    
    layout = [
        [sg.Column([
            [sg.Text('Valorant Aim Tracker', font=('Arial', 18, 'bold'))],
        ], vertical_alignment='top', expand_x=True)],
        
        [sg.HSeparator()],
        
        # Video Selection
        [sg.Text('VIDEO FILE', font=('Arial', 12, 'bold'))],
        [sg.Text('Select video:'), sg.InputText(key='VIDEO_PATH', disabled=True, size=(40, 1)), 
         sg.FileBrowse(initial_folder=VIDEO_FOLDER, file_types=(('Video Files', '*.mp4 *.avi *.mov'),))],
        
        [sg.HSeparator()],
        
        # User Selection
        [sg.Text('PLAYER PROFILE', font=('Arial', 12, 'bold'))],
        [sg.Text('Select Player:'), sg.Combo(user_list, key='USER_SELECT', size=(20, 1), readonly=True),
         sg.Button('New User', size=(10, 1))],
        
        # User Info Display
        [sg.Column([
            [sg.Text('Player:', font=('Arial', 10, 'bold')), sg.Text('', key='USER_NAME', size=(25, 1))],
            [sg.Text('Tag:', font=('Arial', 10, 'bold')), sg.Text('', key='USER_TAG', size=(25, 1))],
        ], vertical_alignment='top')],
        
        [sg.HSeparator()],
        
        # Match Details
        [sg.Text('MATCH DETAILS', font=('Arial', 12, 'bold'))],
        [
            sg.Column([
                [sg.Text('Map:'), sg.Combo(['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze', 
                                            'Fracture', 'Pearl', 'Lotus', 'Sunset', 'Abyss'], key='MAP', size=(15, 1))],
                [sg.Text('Result:'), sg.Combo(['Win', 'Loss'], key='RESULT', size=(15, 1))],
                [sg.Text('Score Line:'), sg.InputText(key='SCORELINE', default_text='e.g., 13-7', size=(15, 1))],
                [sg.Text('Headshot %:'), sg.InputText(key='HS_PERCENT', default_text='e.g., 45.2', size=(15, 1))],
            ], vertical_alignment='top'),
            sg.Column([
                [sg.Text('Kills:'), sg.InputText(key='KILLS', size=(8, 1)),
                 sg.Text('Deaths:'), sg.InputText(key='DEATHS', size=(8, 1)),
                 sg.Text('Assists:'), sg.InputText(key='ASSISTS', size=(8, 1))],
                [sg.Text('ACS:'), sg.InputText(key='ACS', default_text='e.g., 245.3', size=(15, 1))],
                [sg.Text('Notes:'), sg.InputText(key='NOTES', default_text='Optional notes...', size=(35, 1))],
            ], vertical_alignment='top')
        ],
        
        [sg.HSeparator()],
        
        # Action Buttons
        [sg.Button('Analyze Video', size=(15, 2), button_color=('white', 'green')),
         sg.Button('View Results', size=(15, 2)),
         sg.Button('Exit', size=(15, 2), button_color=('white', 'red'))],
        
        [sg.Text('Status: Ready', key='STATUS', text_color='white', background_color='darkgreen', size=(60, 1))],
    ]
    
    window = sg.Window('Valorant Aim Tracker', layout, finalize=True)
    
    return window, users

def main():
    window, users = create_main_window()
    
    while True:
        event, values = window.read()
        
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        
        # New User Button
        if event == 'New User':
            new_user = create_user_window()
            if new_user:
                player_name = new_user['name']
                # Handle duplicate names by adding counter
                counter = 1
                original_name = player_name
                while player_name in users:
                    player_name = f"{original_name} ({counter})"
                    counter += 1
                
                users[player_name] = new_user
                save_users(users)
                
                # Update dropdown
                user_list = list(users.keys())
                window['USER_SELECT'].update(values=user_list)
                window['STATUS'].update(f'Status: User "{player_name}" created successfully!', 
                                       background_color='darkgreen')
                sg.popup_ok(f'User "{player_name}" created successfully!')
        
        # User Selection Changed
        if event == 'USER_SELECT' and values['USER_SELECT']:
            selected_user = values['USER_SELECT']
            user_data = users[selected_user]
            window['USER_NAME'].update(user_data['name'])
            window['USER_TAG'].update(user_data['tag'])
        
        # Analyze Video Button
        if event == 'Analyze Video':
            # Validate inputs
            if not values['VIDEO_PATH']:
                sg.popup_error('Please select a video file')
                continue
            
            if not values['USER_SELECT']:
                sg.popup_error('Please select or create a player profile')
                continue
            
            if not values['MAP']:
                sg.popup_error('Please select a map')
                continue
            
            if not values['RESULT']:
                sg.popup_error('Please select Win or Loss')
                continue
            
            # Validate numeric inputs
            try:
                kills = int(values['KILLS']) if values['KILLS'] else 0
                deaths = int(values['DEATHS']) if values['DEATHS'] else 0
                assists = int(values['ASSISTS']) if values['ASSISTS'] else 0
                hs_percent = float(values['HS_PERCENT']) if values['HS_PERCENT'] else 0
                acs = float(values['ACS']) if values['ACS'] else 0
            except ValueError:
                sg.popup_error('Please enter valid numbers for K/D/A, HS%, and ACS')
                continue
            
            # Show loading window
            window['STATUS'].update('Status: Analyzing video... This may take a few minutes.', 
                                   background_color='orange')
            window.refresh()
            
            try:
                # Run analysis
                video_path = values['VIDEO_PATH']
                selected_user = values['USER_SELECT']
                
                num_cores = mp.cpu_count()
                
                elapsed, analysis_results = detect_enemies_in_video(video_path, max_detected_frames_to_display=25, num_workers=4)
                
                # Save match data
                match_data = {
                    'map': values['MAP'],
                    'result': values['RESULT'],
                    'scoreline': values['SCORELINE'],
                    'kills': kills,
                    'deaths': deaths,
                    'assists': assists,
                    'hs_percent': hs_percent,
                    'acs': acs,
                    'notes': values['NOTES'],
                    'video_path': video_path,
                    'analysis_time': elapsed
                }
                
                # Add analysis results to match data
                if analysis_results:
                    match_data['reaction_times'] = analysis_results['reaction_times']
                    match_data['avg_reaction_time'] = analysis_results['avg_reaction_time']
                    match_data['crosshair_placements'] = analysis_results['crosshair_placements']
                    match_data['avg_crosshair_placement'] = analysis_results['avg_crosshair_placement']
                    match_data['total_engagements'] = analysis_results['total_engagements']
                    match_data['engagements_with_shots'] = analysis_results['engagements_with_shots']
                
                users[selected_user]['matches'].append(match_data)
                save_users(users)
                
                # Show results to user
                results_msg = f'Analysis complete!\n\nTime: {elapsed:.2f}s\n\nMatch data saved for {selected_user}'
                if analysis_results:
                    results_msg += f'\n\n--- ANALYSIS RESULTS ---'
                    results_msg += f'\nTotal Engagements: {analysis_results["total_engagements"]}'
                    results_msg += f'\nEngagements with Shots: {analysis_results["engagements_with_shots"]}'
                    results_msg += f'\nAverage Reaction Time: {analysis_results["avg_reaction_time"]:.3f}s'
                    results_msg += f'\nMost Common Crosshair Placement: {analysis_results["avg_crosshair_placement"]}'
                
                window['STATUS'].update(f'Status: Analysis complete! ({elapsed:.2f}s)', 
                                       background_color='darkgreen')
                sg.popup_ok(results_msg)
                
            except Exception as e:
                window['STATUS'].update(f'Status: Error - {str(e)}', background_color='darkred')
                sg.popup_error(f'Error during analysis:\n{str(e)}')
        
        # View Results Button
        if event == 'View Results':
            if not values['USER_SELECT']:
                sg.popup_error('Please select a player profile')
                continue
            
            selected_user = values['USER_SELECT']
            user_data = users[selected_user]
            
            if not user_data['matches']:
                sg.popup_info(f'No matches recorded for {selected_user}')
                continue
            
            # Create results window
            results_layout = [
                [sg.Text(f'Matches for {user_data["name"]} ({user_data["tag"]})', font=('Arial', 14, 'bold'))],
                [sg.HSeparator()],
            ]
            
            for i, match in enumerate(user_data['matches'], 1):
                results_layout.append([
                    sg.Text(f'Match {i}: {match["result"].upper()} on {match["map"]} | {match["scoreline"]} | K/D/A: {match["kills"]}/{match["deaths"]}/{match["assists"]} | HS: {match["hs_percent"]}% | ACS: {match["acs"]}')
                ])
                if match['notes']:
                    results_layout.append([sg.Text(f'  Notes: {match["notes"]}', text_color='gray')])
                
                # Add analysis results if available
                if 'avg_reaction_time' in match:
                    results_layout.append([sg.Text(f'  📊 Analysis:', font=('Arial', 10, 'bold'))])
                    results_layout.append([sg.Text(f'     • Engagements: {match.get("total_engagements", 0)} ({match.get("engagements_with_shots", 0)} with shots)', text_color='lightblue')])
                    results_layout.append([sg.Text(f'     • Avg Reaction Time: {match.get("avg_reaction_time", 0):.3f}s', text_color='lightblue')])
                    results_layout.append([sg.Text(f'     • Crosshair Placement: {match.get("avg_crosshair_placement", "N/A")}', text_color='lightblue')])
                
                results_layout.append([sg.Text('')])  # Spacing
            
            results_layout.append([sg.Button('Close')])
            
            results_window = sg.Window('Match Results', results_layout)
            while True:
                r_event, r_values = results_window.read()
                if r_event == sg.WINDOW_CLOSED or r_event == 'Close':
                    results_window.close()
                    break
    
    window.close()

if __name__ == '__main__':
    main()
