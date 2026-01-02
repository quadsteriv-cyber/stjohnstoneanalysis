# ----------------------------------------------------------------------
# ⚽ Advanced Multi-Position Player Analysis App v12.0 (Canonical Seasons) ⚽
#
# Changes in this version:
# - Implemented canonical season logic to group seasons by their end year (e.g., 2024/25 + 2025).
# - Added helper functions to support the new season logic.
# - Maintained all previous features (GK section, Fullscreen Radars, etc.).
# ----------------------------------------------------------------------

# --- 1. IMPORTS ---
import streamlit as st
import requests
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from datetime import date

# Plotly + HTML component for legend-hover interactivity
import plotly.graph_objects as go
import plotly.io as pio
import uuid
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# --- 2. APP CONFIGURATION ---



def main():
    """Run the Streamlit UI. Safe to import this module without side-effects."""
    import streamlit as st

    st.set_page_config(
        page_title="Advanced Player Analysis",
        page_icon="⚽",
        layout="wide"
    )
    # Initialize session state variables
    st.write("Initializing app...")  # DEBUG: Confirm we're in main()
    if 'comp_selections' not in st.session_state:
        st.session_state.comp_selections = {"league": None, "season": None, "team": None, "player": None}
    if 'comparison_players' not in st.session_state:
        st.session_state.comparison_players = []
    if 'radar_players' not in st.session_state:
        st.session_state.radar_players = []
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'target_player' not in st.session_state:
        st.session_state.target_player = None
    if 'detected_archetype' not in st.session_state:
        st.session_state.detected_archetype = None
    if 'dna_df' not in st.session_state:
        st.session_state.dna_df = None
    if 'matches' not in st.session_state:
        st.session_state.matches = None
    if 'unknown_age_count' not in st.session_state:
        st.session_state.unknown_age_count = 0
    if 'analysis_pos' not in st.session_state:
        st.session_state.analysis_pos = None

    # --- 3. CORE & POSITIONAL CONFIGURATIONS ---
    import os
    import streamlit as st

    USERNAME = os.getenv("STATSBOMB_USERNAME")
    PASSWORD = os.getenv("STATSBOMB_PASSWORD")

    if not USERNAME or not PASSWORD:
        st.error("StatsBomb credentials not found. Check Codespaces secrets.")
        st.stop()

    LEAGUE_NAMES = {
        4: "League One", 5: "League Two", 51: "Premiership", 65: "National League",
        76: "Liga", 78: "1. HNL", 89: "USL Championship", 106: "Veikkausliiga",
        107: "Premier Division", 129: "Championnat National", 166: "Premier League 2 Division One",
        179: "3. Liga", 260: "1st Division", 1035: "First Division B", 1385: "Championship",
        1442: "1. Division", 1581: "2. Liga", 1607: "Úrvalsdeild", 1778: "First Division",
        1848: "I Liga", 1865: "First League"
    }

    COMPETITION_SEASONS = {
        4: [235, 281, 317, 318],
        5: [235, 281, 317, 318],
        51: [235, 281, 317, 318],
        65: [281, 318],
        76: [317, 318],
        78: [317, 318],
        89: [106, 107, 282, 315],
        106: [315],
        107: [106, 107, 282, 315],
        129: [317, 318],
        166: [318],
        179: [317, 318],
        260: [317, 318],
        1035: [317, 318],
        1385: [235, 281, 317, 318],
        1442: [107, 282, 315],
        1581: [317, 318],
        1607: [315],
        1778: [282, 315],
        1848: [281, 317, 318],
        1865: [318]
    }

    DOMESTIC_LEAGUE_IDS = [4, 5, 51, 65, 1385, 166]
    SCOTTISH_LEAGUE_IDS = [51]

    # Archetype definitions
    STRIKER_ARCHETYPES = {
        "Poacher (Fox in the Box)": {
            "description": "A clinical finisher who thrives in the penalty area with instinctive movement and a high shot volume. Minimal involvement in build-up play outside the final third. They prioritize shooting over passing.",
            "identity_metrics": ['npg_90', 'np_xg_90', 'np_shots_90', 'touches_inside_box_90', 'conversion_ratio', 'np_xg_per_shot', 'shot_touch_ratio', 'op_xgchain_90'],
            "key_weight": 1.7
        },
        "Target Man": {
            "description": "A physically dominant forward with a strong aerial presence, excels at holding up the ball and bringing teammates into play. They are a focal point for long balls and physical duels.",
            "identity_metrics": ['aerial_wins_90', 'aerial_ratio', 'fouls_won_90', 'op_xgbuildup_90', 'carries_90', 'touches_inside_box_90', 'long_balls_90', 'passing_ratio'],
            "key_weight": 1.6
        },
        "Complete Forward": {
            "description": "A well-rounded striker capable of doing everything: finishing, dribbling, linking up play, and making intelligent runs. A central figure in both goal-scoring and chance creation.",
            "identity_metrics": ['npg_90', 'key_passes_90', 'dribbles_90', 'deep_progressions_90', 'op_xgbuildup_90', 'aerial_wins_90', 'op_xgchain_90', 'npxgxa_90'],
            "key_weight": 1.6
        },
        "False 9": {
            "description": "A forward who drops deep into midfield to link play, acting more like a playmaker than a traditional striker. They possess excellent technical skills, vision, and a high xG buildup contribution.",
            "identity_metrics": ['op_xgbuildup_90', 'key_passes_90', 'through_balls_90', 'dribbles_90', 'carries_90', 'xa_90', 'forward_pass_proportion', 'passing_ratio'],
            "key_weight": 1.5
        },
        "Advanced Forward": {
            "description": "A pacey forward who primarily makes runs in behind the defensive line. They thrive on through balls and quick transitions, focusing on getting into dangerous areas to shoot.",
            "identity_metrics": ['deep_progressions_90', 'through_balls_90', 'np_shots_90', 'touches_inside_box_90', 'npg_90', 'np_xg_90', 'dribbles_90', 'npxgxa_90'],
            "key_weight": 1.6
        },
        "Pressing Forward": {
            "description": "A high-energy striker whose main defensive contribution is to harass and pressure opposition defenders. They have a high work rate and actively participate in winning the ball back.",
            "identity_metrics": ['pressures_90', 'pressure_regains_90', 'counterpressures_90', 'aggressive_actions_90', 'padj_tackles_90', 'fouls_90', 'fhalf_pressures_90', 'fhalf_counterpressures_90'],
            "key_weight": 1.5
        },
    }

    # Radar metrics
    STRIKER_RADAR_METRICS = {
        'finishing': {
            'name': 'Finishing', 'color': '#D32F2F',
            'metrics': {
                'npg_90': 'Non-Penalty Goals', 'np_xg_90': 'Non-Penalty xG',
                'np_shots_90': 'Shots p90', 'conversion_ratio': 'Shot Conversion %',
                'np_xg_per_shot': 'Avg. Shot Quality', 'touches_inside_box_90': 'Touches in Box p90'
            }
        },
        'box_presence': {
            'name': 'Box Presence', 'color': '#AF1D1D',
            'metrics': {
                'touches_inside_box_90': 'Touches in Box p90',
                'passes_inside_box_90': 'Passes in Box p90',
                'positive_outcome_90': 'Positive Outcomes p90',
                'shot_touch_ratio': 'Shot/Touch %',
                'op_passes_into_box_90': 'Passes into Box p90',
                'np_xg_per_shot': 'Avg. Shot Quality'
            }
        },
        'creation': {
            'name': 'Creation & Link-Up', 'color': '#FF6B35',
            'metrics': {
                'key_passes_90': 'Key Passes p90', 'xa_90': 'xA p90',
                'op_passes_into_box_90': 'Passes into Box p90', 'through_balls_90': 'Through Balls p90',
                'op_xgbuildup_90': 'xG Buildup p90', 'passing_ratio': 'Pass Completion %'
            }
        },
        'dribbling': {
            'name': 'Dribbling & Carrying', 'color': '#9C27B0',
            'metrics': {
                'dribbles_90': 'Successful Dribbles p90', 'dribble_ratio': 'Dribble Success %',
                'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
                'turnovers_90': 'Ball Security (Inv)', 'deep_progressions_90': 'Deep Progressions p90'
            }
        },
        'aerial': {
            'name': 'Aerial Prowess', 'color': '#607D8B',
            'metrics': {
                'aerial_wins_90': 'Aerial Duels Won p90', 'aerial_ratio': 'Aerial Win %',
                'aggressive_actions_90': 'Aggressive Actions p90', 'challenge_ratio': 'Defensive Duel Win %',
                'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
                'fouls_won_90': 'Fouls Won p90'
            }
        },
        'defensive': {
            'name': 'Defensive Contribution', 'color': '#4CAF50',
            'metrics': {
                'pressures_90': 'Pressures p90', 'pressure_regains_90': 'Pressure Regains p90',
                'counterpressures_90': 'Counterpressures p90', 'aggressive_actions_90': 'Aggressive Actions',
                'padj_tackles_90': 'P.Adj Tackles p90', 'dribbled_past_90': 'Times Dribbled Past p90'
            }
        }
    }

    WINGER_ARCHETYPES = {
        "Goal-Scoring Winger": {
            "description": "A winger focused on cutting inside to shoot and score goals, often functioning as a wide forward. They have a high goal threat and strong dribbling ability.",
            "identity_metrics": ['npg_90', 'np_xg_90', 'np_shots_90', 'touches_inside_box_90', 'np_xg_per_shot', 'dribbles_90', 'over_under_performance_90', 'npxgxa_90', 'op_passes_into_box_90'],
            "key_weight": 1.6
        },
        "Creative Playmaker": {
            "description": "A winger who creates chances for others through key passes, crosses, and assists. They are a primary source of creativity from wide areas and often have a high xG buildup contribution.",
            "identity_metrics": ['xa_90', 'key_passes_90', 'op_passes_into_box_90', 'through_balls_90', 'op_xgbuildup_90', 'deep_progressions_90', 'crosses_90', 'dribbles_90', 'fouls_won_90'],
            "key_weight": 1.5
        },
        "Traditional Winger": {
            "description": "A winger who focuses on providing width and stretching the opposition defense. Their primary actions are dribbling down the line and delivering crosses into the box.",
            "identity_metrics": ['crosses_90', 'crossing_ratio', 'dribbles_90', 'carry_length', 'deep_progressions_90', 'fouls_won_90', 'op_passes_into_box_90', 'turnovers_90'],
            "key_weight": 1.5
        },
        "Inverted Winger": {
            "description": "A winger who plays on the opposite flank of their strong foot, allowing them to cut inside and create. They are defined by a high volume of successful dribbles and a strong role in ball progression and attacking buildup.",
            "identity_metrics": ['dribbles_90', 'dribble_ratio', 'carries_90', 'carry_length', 'deep_progressions_90', 'op_xgbuildup_90', 'op_passes_into_box_90', 'xa_90'],
            "key_weight": 1.6
        }
    }

    WINGER_RADAR_METRICS = {
        'goal_threat': {
            'name': 'Goal Threat', 'color': '#D32F2F',
            'metrics': {
                'npg_90': 'Non-Penalty Goals', 'np_xg_90': 'Non-Penalty xG',
                'np_shots_90': 'Shots p90', 'touches_inside_box_90': 'Touches in Box p90',
                'conversion_ratio': 'Shot Conversion %', 'np_xg_per_shot': 'Avg. Shot Quality'
            }
        },
        'creation': {
            'name': 'Chance Creation', 'color': '#FF6B35',
            'metrics': {
                'key_passes_90': 'Key Passes p90', 'xa_90': 'xA p90',
                'op_passes_into_box_90': 'Passes into Box p90', 'through_balls_90': 'Through Balls p90',
                'op_xgbuildup_90': 'xG Buildup p90', 'passing_ratio': 'Pass Completion %'
            }
        },
        'progression': {
            'name': 'Dribbling & Progression', 'color': '#9C27B0',
            'metrics': {
                'dribbles_90': 'Successful Dribbles p90', 'dribble_ratio': 'Dribble Success %',
                'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
                'deep_progressions_90': 'Deep Progressions p90', 'fouls_won_90': 'Fouls Won p90'
            }
        },
        'crossing': {
            'name': 'Crossing Profile', 'color': '#00BCD4',
            'metrics': {
                'crosses_90': 'Completed Crosses p90', 'crossing_ratio': 'Cross Completion %',
                'box_cross_ratio': '% of Box Passes that are Crosses', 'op_passes_into_box_90': 'Passes into Box p90',
                'key_passes_90': 'Key Passes p90', 'xa_90': 'xA p90'
            }
        },
        'defensive': {
            'name': 'Defensive Work Rate', 'color': '#4CAF50',
            'metrics': {
                'pressures_90': 'Pressures p90', 'pressure_regains_90': 'Pressure Regains p90',
                'padj_tackles_90': 'P.Adj Tackles p90', 'padj_interceptions_90': 'P.Adj Interceptions p90',
                'dribbled_past_90': 'Times Dribbled Past p90', 'aggressive_actions_90': 'Aggressive Actions'
            }
        },
        'duels': {
            'name': 'Duels & Security', 'color': '#607D8B',
            'metrics': {
                'aerial_wins_90': 'Aerial Duels Won p90', 'aerial_ratio': 'Aerial Win %',
                'challenge_ratio': 'Defensive Duel Win %', 'fouls_won_90': 'Fouls Won p90',
                'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
                'turnovers_90': 'Ball Security (Inv)'
            }
        }
    }

    CM_ARCHETYPES = {
        "Deep-Lying Playmaker (Regista)": {
            "description": "A midfielder who dictates tempo from deep positions, excelling in progressive passing and ball distribution to start attacks. They are the team's engine from the defensive half.",
            "identity_metrics": ['op_xgbuildup_90', 'long_balls_90', 'long_ball_ratio', 'forward_pass_proportion', 'passing_ratio', 'through_balls_90', 'op_f3_passes_90', 'carries_90'],
            "key_weight": 1.6
        },
        "Box-to-Box Midfielder (B2B)": {
            "description": "A high-energy midfielder who covers large vertical space on the pitch, contributing heavily in both attack and defense. They are involved in ball progression, tackling, and late runs into the box.",
            "identity_metrics": ['deep_progressions_90', 'carries_90', 'padj_tackles_and_interceptions_90', 'pressures_90', 'npg_90', 'touches_inside_box_90', 'op_xgchain_90', 'offensive_duels_90'],
            "key_weight": 1.6
        },
        "Ball-Winning Midfielder (Destroyer)": {
            "description": "A defensive-minded midfielder who breaks up opposition attacks, screens the defense, and wins possession. They are defined by their tenacity and high volume of defensive actions.",
            "identity_metrics": ['padj_tackles_90', 'padj_interceptions_90', 'pressure_regains_90', 'challenge_ratio', 'aggressive_actions_90', 'fouls_90', 'dribbled_past_90'],
            "key_weight": 1.6
        },
        "Advanced Playmaker (Mezzala)": {
            "description": "A creative midfielder who operates in the half-spaces and creates chances in advanced zones. They are excellent dribblers and key passers who often make runs into the final third.",
            "identity_metrics": ['xa_90', 'key_passes_90', 'op_passes_into_box_90', 'through_balls_90', 'dribbles_90', 'np_shots_90', 'op_xgbuildup_90', 'deep_progressions_90'],
            "key_weight": 1.5
        },
        "Holding Midfielder (Anchor)": {
            "description": "A conservative midfielder who protects the backline and distributes the ball safely and efficiently. They are defined by their positional discipline and high pass completion rate.",
            "identity_metrics": ['padj_interceptions_90', 'passing_ratio', 'op_xgbuildup_90', 'pressures_90', 'challenge_ratio', 'turnovers_90', 'padj_clearances_90', 's_pass_length'],
            "key_weight": 1.5
        },
        "Attacking Midfielder (8.5 Role)": {
            "description": "An aggressive, goal-oriented midfielder who operates closer to the opposition box, focusing on final-third involvement and attacking output, similar to a second striker.",
            "identity_metrics": ['npg_90', 'np_xg_90', 'xa_90', 'key_passes_90', 'touches_inside_box_90', 'np_shots_90', 'op_passes_into_box_90', 'dribbles_90'],
            "key_weight": 1.6
        }
    }


    CM_RADAR_METRICS = {
        'defending': {
            'name': 'Defensive Actions', 'color': '#D32F2F',
            'metrics': {
                'padj_tackles_and_interceptions_90': 'P.Adj Tackles+Ints',
                'challenge_ratio': 'Defensive Duel Win %',
                'dribbled_past_90': 'Times Dribbled Past p90',
                'aggressive_actions_90': 'Aggressive Actions',
                'pressures_90': 'Pressures p90'
            }
        },
        'duels': {
            'name': 'Duels & Physicality', 'color': '#AF1D1D',
            'metrics': {
                'aerial_wins_90': 'Aerial Duels Won', 'aerial_ratio': 'Aerial Win %',
                'fouls_won_90': 'Fouls Won', 'challenge_ratio': 'Defensive Duel Win %',
                'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
                'aggressive_actions_90': 'Aggressive Actions'
            }
        },
        'passing': {
            'name': 'Passing & Distribution', 'color': '#0066CC',
            'metrics': {
                'passing_ratio': 'Pass Completion %', 'forward_pass_proportion': 'Forward Pass %',
                'long_balls_90': 'Long Balls p90', 'long_ball_ratio': 'Long Ball Accuracy %',
                'op_xgbuildup_90': 'xG Buildup p90'
            }
        },
        'creation': {
            'name': 'Creativity & Creation', 'color': '#FF6B35',
            'metrics': {
                'key_passes_90': 'Key Passes p90', 'xa_90': 'xA p90',
                'through_balls_90': 'Through Balls p90', 'op_xgbuildup_90': 'xG Buildup p90',
                'op_passes_into_box_90': 'Passes into Box p90'
            }
        },
        'progression': {
            'name': 'Ball Progression', 'color': '#4CAF50',
            'metrics': {
                'deep_progressions_90': 'Deep Progressions', 'carries_90': 'Ball Carries p90',
                'carry_length': 'Avg. Carry Length', 'dribbles_90': 'Successful Dribbles',
                'dribble_ratio': 'Dribble Success %'
            }
        },
        'attacking': {
            'name': 'Attacking Output', 'color': '#9C27B0',
            'metrics': {
                'npg_90': 'Non-Penalty Goals', 'np_xg_90': 'Non-Penalty xG',
                'np_shots_90': 'Shots p90', 'touches_inside_box_90': 'Touches in Box',
                'np_xg_per_shot': 'Avg. Shot Quality'
            }
        }
    }

    FULLBACK_ARCHETYPES = {
        "Attacking Fullback": {
            "description": "An offensive-minded full-back with high attacking output, including crosses, key passes, and deep forward runs into the final third to create chances.",
            "identity_metrics": ['xa_90', 'crosses_90', 'op_passes_into_box_90', 'deep_progressions_90', 'key_passes_90', 'op_xgbuildup_90', 'dribbles_90', 'fouls_won_90'],
            "key_weight": 1.5
        },
        "Defensive Fullback": {
            "description": "A traditional full-back with a solid defensive foundation, focusing on preventing attacks through tackling, interceptions, and aerial duels.",
            "identity_metrics": ['padj_tackles_and_interceptions_90', 'challenge_ratio', 'aggressive_actions_90', 'pressures_90', 'aerial_wins_90', 'aerial_ratio', 'dribbled_past_90', 'padj_clearances_90'],
            "key_weight": 1.5
        },
        "Modern Wingback": {
            "description": "A high-energy, all-action player who contributes in both defense and attack. They possess high stamina and cover large distances, excelling in both progression and defensive work rate.",
            "identity_metrics": ['deep_progressions_90', 'crosses_90', 'dribbles_90', 'padj_tackles_and_interceptions_90', 'pressures_90', 'xa_90', 'pressure_regains_90', 'op_xgbuildup_90'],
            "key_weight": 1.6
        },
        "Inverted Fullback": {
            "description": "A fullback who moves into central midfield areas when their team has possession, excelling at linking play and progressive passing from deep zones.",
            "identity_metrics": ['passing_ratio', 'deep_progressions_90', 'op_xgbuildup_90', 'carries_90', 'forward_pass_proportion', 'padj_tackles_90', 'padj_interceptions_90', 'dribble_ratio'],
            "key_weight": 1.7
        }
    }

    FULLBACK_RADAR_METRICS = {
        'defensive_actions': {
            'name': 'Defensive Actions', 'color': '#00BCD4',
            'metrics': {
                'padj_tackles_and_interceptions_90': 'P.Adj Tackles+Ints p90',
                'challenge_ratio': 'Defensive Duel Win %',
                'dribbled_past_90': 'Times Dribbled Past p90',
                'pressures_90': 'Pressures p90',
                'aggressive_actions_90': 'Aggressive Actions p90'
            }
        },
        'duels': {
            'name': 'Duels', 'color': '#008294',
            'metrics': {
                'aerial_wins_90': 'Aerial Duels Won p90', 'aerial_ratio': 'Aerial Win %',
                'aggressive_actions_90': 'Aggressive Actions p90', 'fouls_won_90': 'Fouls Won p90',
                'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length'
            }
        },
        'progression_creation': {
            'name': 'Progression & Creation', 'color': '#FF6B35',
            'metrics': {
                'deep_progressions_90': 'Deep Progressions p90', 'carries_90': 'Ball Carries p90',
                'dribbles_90': 'Successful Dribbles p90', 'xa_90': 'xA p90',
                'op_passes_into_box_90': 'Passes into Box p90'
            }
        },
        'crossing': {
            'name': 'Crossing', 'color': '#FFA735',
            'metrics': {
                'crosses_90': 'Completed Crosses p90', 'crossing_ratio': 'Cross Completion %',
                'box_cross_ratio': '% of Box Passes that are Crosses', 'key_passes_90': 'Key Passes p90'
            }
        },
        'passing': {
            'name': 'Passing & Buildup', 'color': '#9C27B0',
            'metrics': {
                'passing_ratio': 'Pass Completion %', 'op_xgbuildup_90': 'xG Buildup p90',
                'key_passes_90': 'Key Passes p90', 'forward_pass_proportion': 'Forward Pass %'
            }
        },
        'work_rate': {
            'name': 'Work Rate & Security', 'color': '#4CAF50',
            'metrics': {
                'pressures_90': 'Pressures p90', 'pressure_regains_90': 'Pressure Regains p90',
                'turnovers_90': 'Ball Security (Inv)', 'dribbled_past_90': 'Times Dribbled Past p90'
            }
        }
    }

    CB_ARCHETYPES = {
        "Ball-Playing Defender": {
            "description": "A defender comfortable in possession, who initiates attacks from the back with progressive passing, long balls, and carries into midfield. They are defined by their on-ball ability.",
            "identity_metrics": ['op_xgbuildup_90', 'passing_ratio', 'long_balls_90', 'long_ball_ratio', 'forward_pass_proportion', 'carries_90', 'deep_progressions_90', 'op_f3_passes_90'],
            "key_weight": 1.5
        },
        "Stopper": {
            "description": "An aggressive defender who steps out to challenge attackers and win the ball high up the pitch. They rely on their physical and combative qualities to break up play before it reaches the box.",
            "identity_metrics": ['aggressive_actions_90', 'padj_tackles_90', 'challenge_ratio', 'pressures_90', 'aerial_wins_90', 'fouls_90', 'pressure_regains_90', 'dribbled_past_90'],
            "key_weight": 1.6
        },
        "Covering Defender": {
            "description": "A defender who reads the game well and relies on superior positioning and interceptions to sweep up behind the defensive line. They are defined by their intelligence and ability to recover the ball with minimal duels.",
            "identity_metrics": ['padj_interceptions_90', 'padj_clearances_90', 'dribbled_past_90', 'pressure_regains_90', 'aerial_ratio', 'passing_ratio', 'turnovers_90', 'average_x_defensive_action'],
            "key_weight": 1.5
        },
        "No-Nonsense Defender": {
            "description": "A physical defender who prioritizes safety and direct action. They excel at aerial duels, clearances, and tackling, with minimal involvement in attacking buildup or ball progression.",
            "identity_metrics": ['padj_clearances_90', 'aerial_wins_90', 'aerial_ratio', 'padj_tackles_90', 'aggressive_actions_90', 'op_xgbuildup_90', 'passing_ratio', 'turnovers_90'],
            "key_weight": 1.7
        }
    }

    CB_RADAR_METRICS = {
        'ground_defending': {
            'name': 'Ground Duels', 'color': '#D32F2F',
            'metrics': {
                'padj_tackles_90': 'PAdj Tackles', 'challenge_ratio': 'Challenge Success %',
                'aggressive_actions_90': 'Aggressive Actions', 'pressures_90': 'Pressures p90'
            }
        },
        'aerial_duels': {
            'name': 'Aerial Duels & Clearances', 'color': '#4CAF50',
            'metrics': {
                'aerial_wins_90': 'Aerial Duels Won', 'aerial_ratio': 'Aerial Win %',
                'padj_clearances_90': 'PAdj Clearances', 'fouls_won_90': 'Fouls Won',
                'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length'
            }
        },
        'passing_distribution': {
            'name': 'Passing & Distribution', 'color': '#0066CC',
            'metrics': {
                'passing_ratio': 'Pass Completion %', 'pass_length': 'Avg. Pass Length',
                'long_balls_90': 'Long Balls p90', 'long_ball_ratio': 'Long Ball Accuracy %',
                'forward_pass_proportion': 'Forward Pass %'
            }
        },
        'ball_progression': {
            'name': 'Ball Progression', 'color': '#FFC107',
            'metrics': {
                'carries_90': 'Ball Carries p90', 'carry_length': 'Avg. Carry Length',
                'deep_progressions_90': 'Deep Progressions', 'op_xgbuildup_90': 'xG Buildup p90'
            }
        },
        'defensive_positioning': {
            'name': 'Defensive Positioning', 'color': '#00BCD4',
            'metrics': {
                'padj_interceptions_90': 'PAdj Interceptions', 'dribbled_past_90': 'Times Dribbled Past p90',
                'pressure_regains_90': 'Pressure Regains', 'turnovers_90': 'Ball Security (Inv)'
            }
        },
        'on_ball_security': {
            'name': 'On-Ball Security', 'color': '#607D8B',
            'metrics': {
                'turnovers_90': 'Ball Security (Inv)', 'op_xgbuildup_90': 'xG Buildup p90',
                'fouls_90': 'Fouls Committed', 'passing_ratio': 'Pass Completion %'
            }
        }
    }

    GK_ARCHETYPES = {
        "Sweeper-Keeper": {
            "description": "A proactive goalkeeper who operates outside the penalty area, intercepting through balls, and participating in the team's buildup play with their feet.",
            "identity_metrics": [
                'avg_pass_length', 'long_ball_ratio', 'op_xgbuildup_90', 'defensive_actions_outside_box_90',
                'padj_interceptions_90', 'carries_90', 'passing_ratio'
            ],
            "key_weight": 1.6
        },
        "Shot-Stopper": {
            "description": "A traditional goalkeeper who excels at making saves and commanding the penalty box. Their primary strengths are reflexes, positioning, and preventing goals.",
            "identity_metrics": [
                'psxg_net_90', 'save_ratio', 'op_saves_90', 'aerial_ratio',
                'aerial_wins_90', 'padj_clearances_90', 'penalty_save_ratio'
            ],
            "key_weight": 1.6
        }
    }

    GK_RADAR_METRICS = {
        'shot_stopping': {
            'name': 'Shot-Stopping', 'color': '#D32F2F',
            'metrics': {
                'psxg_net_90': 'Goals Prevented p90',
                'save_ratio': 'Save %',
                'op_saves_90': 'Saves from Open Play p90',
                'penalty_save_ratio': 'Penalty Save %',
                'cross_claim_ratio': 'Cross Claim %'
            }
        },
        'aerial_command': {
            'name': 'Aerial Command', 'color': '#607D8B',
            'metrics': {
                'aerial_wins_90': 'Aerial Duels Won p90',
                'aerial_ratio': 'Aerial Win %',
                'cross_claim_ratio': 'Cross Claim %',
                'padj_clearances_90': 'P.Adj Clearances p90',
                'avg_x_defensive_action': 'Avg. Defensive Action Distance'
            }
        },
        'distribution': {
            'name': 'Distribution & Passing', 'color': '#0066CC',
            'metrics': {
                'passing_ratio': 'Pass Completion %',
                'long_ball_ratio': 'Long Ball Accuracy %',
                'avg_pass_length': 'Avg. Pass Length',
                'op_xgbuildup_90': 'xG Buildup p90',
                'launches_ratio': 'Launch Completion % (>=40yds)'
            }
        },
        'sweeping': {
            'name': 'Sweeping Actions', 'color': '#4CAF50',
            'metrics': {
                'defensive_actions_outside_box_90': 'Def. Actions Outside Box p90',
                'avg_x_defensive_action': 'Avg. Defensive Action Distance',
                'padj_interceptions_90': 'P.Adj Interceptions p90',
                'pressures_90': 'Pressures p90'
            }
        }
    }

    POSITIONAL_CONFIGS = {
        "Goalkeeper": {"archetypes": GK_ARCHETYPES, "radars": GK_RADAR_METRICS, "positions": ['Goalkeeper']},
        "Fullback": {"archetypes": FULLBACK_ARCHETYPES, "radars": FULLBACK_RADAR_METRICS, "positions":
                     ['Left Back', 'Left Wing Back', 'Right Back', 'Right Wing Back']},
        "Center Back": {"archetypes": CB_ARCHETYPES, "radars": CB_RADAR_METRICS, "positions":
                        ['Centre Back', 'Left Centre Back', 'Right Centre Back']},
        "Center Midfielder": {"archetypes": CM_ARCHETYPES, "radars": CM_RADAR_METRICS, "positions": [
            'Centre Attacking Midfielder', 'Centre Defensive Midfielder', 'Left Centre Midfielder',
            'Left Defensive Midfielder', 'Right Centre Midfielder', 'Right Defensive Midfielder'
        ]},
        "Winger": {"archetypes": WINGER_ARCHETYPES, "radars": WINGER_RADAR_METRICS, "positions": [
            'Left Attacking Midfielder', 'Left Midfielder', 'Left Wing',
            'Right Attacking Midfielder', 'Right Midfielder', 'Right Wing'
        ]},
        "Striker": {"archetypes": STRIKER_ARCHETYPES, "radars": STRIKER_RADAR_METRICS, "positions": [
            'Centre Forward', 'Left Centre Forward', 'Right Centre Forward', 'Secondary Striker'
        ]}
    }


    ALL_METRICS_TO_PERCENTILE = sorted(list(set(
        metric for pos_config in POSITIONAL_CONFIGS.values()
        for archetype in pos_config['archetypes'].values() for metric in archetype['identity_metrics']
    ) | set(
        metric for pos_config in POSITIONAL_CONFIGS.values()
        for radar in pos_config['radars'].values() for metric in radar['metrics'].keys()
    )))

    # --- 4. DATA HANDLING & ANALYSIS FUNCTIONS ---

    @st.cache_resource(ttl=3600)
    def get_all_leagues_data(_auth_credentials):
        """Downloads player statistics from all leagues with improved error handling."""
        all_dfs = []
        successful_loads = 0
        failed_loads = 0

        try:
            test_url = "https://data.statsbombservices.com/api/v4/competitions"
            test_response = requests.get(test_url, auth=_auth_credentials, timeout=30)
            test_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Authentication failed. Please check your username and password. Error: {e}")
            return None

        total_requests = sum(len(season_ids) for season_ids in COMPETITION_SEASONS.values())
        progress_bar = st.progress(0)
        status_text = st.empty()

        current_request = 0

        for league_id, season_ids in COMPETITION_SEASONS.items():
            league_name = LEAGUE_NAMES.get(league_id, f"League {league_id}")

            for season_id in season_ids:
                current_request += 1
                progress = current_request / total_requests
                progress_bar.progress(progress)
                status_text.text(f"Loading {league_name} (Season {season_id})... {current_request}/{total_requests}")

                try:
                    url = f"https://data.statsbombservices.com/api/v1/competitions/{league_id}/seasons/{season_id}/player-stats"
                    response = requests.get(url, auth=_auth_credentials, timeout=60)
                    response.raise_for_status()

                    data = response.json()
                    if not data:
                        failed_loads += 1
                        continue

                    df_league = pd.json_normalize(data)
                    if df_league.empty:
                        failed_loads += 1
                        continue

                    df_league['league_name'] = league_name
                    df_league['competition_id'] = league_id
                    df_league['season_id'] = season_id
                    all_dfs.append(df_league)
                    successful_loads += 1

                except Exception:
                    failed_loads += 1
                    continue

        progress_bar.empty()
        status_text.empty()

        if not all_dfs:
            st.error("Could not load any data from the API. Please check your internet connection and API credentials.")
            return None

        st.success(f"Successfully loaded data from {successful_loads} league/season combinations.")

        try:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            return combined_df
        except Exception as e:
            st.error(f"Error combining datasets: {e}")
            return None

    def get_canonical_season(season_str):
        """
        Intelligently extracts the canonical end year from a season string.
        e.g., '2025' from '2024/2025' and '2025' from '2025'.
        This allows grouping different season formats together.
        """
        try:
            if isinstance(season_str, str) and '/' in season_str:
                return int(season_str.split('/')[1])  # Take the END year
            else:
                return int(season_str)
        except (ValueError, TypeError):
            return 0

    @st.cache_data(ttl=3600)
    def process_data(_raw_data):
        """Processes raw data to calculate ages, position groups, and normalized metrics"""
        if _raw_data is None:
            return None

        df_processed = _raw_data.copy()
        df_processed.columns = [c.replace('player_season_', '') for c in df_processed.columns]

        for col in ['player_name', 'team_name', 'league_name', 'season_name', 'primary_position']:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].str.strip()

        def calculate_age(birth_date_str):
            if pd.isna(birth_date_str): return None
            try:
                birth_date = pd.to_datetime(birth_date_str).date()
                today = date.today()
                return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            except (ValueError, TypeError): return None
        df_processed['age'] = df_processed['birth_date'].apply(calculate_age)

        def get_position_group(primary_position):
            for group, config in POSITIONAL_CONFIGS.items():
                if primary_position in config['positions']:
                    return group
            return None
        df_processed['position_group'] = df_processed['primary_position'].apply(get_position_group)

        if 'padj_tackles_90' in df_processed.columns and 'padj_interceptions_90' in df_processed.columns:
            df_processed['padj_tackles_and_interceptions_90'] = (
                df_processed['padj_tackles_90'] + df_processed['padj_interceptions_90']
            )

        negative_stats = ['turnovers_90', 'dispossessions_90', 'dribbled_past_90', 'fouls_90']

        for metric in ALL_METRICS_TO_PERCENTILE:
            if metric not in df_processed.columns:
                df_processed[metric] = 0
                continue

            df_processed[f'{metric}_pct'] = 0
            df_processed[f'{metric}_z'] = 0.0

            for group, group_df in df_processed.groupby('position_group', dropna=False):
                if group is None or len(group_df) < 5:
                    continue

                metric_series = group_df[metric]

                if metric in negative_stats:
                    ranks = metric_series.rank(pct=True, ascending=True)
                    df_processed.loc[group_df.index, f'{metric}_pct'] = (1 - ranks) * 100
                else:
                    df_processed.loc[group_df.index, f'{metric}_pct'] = metric_series.rank(pct=True) * 100

                scaler = StandardScaler()
                z_scores = scaler.fit_transform(metric_series.values.reshape(-1, 1)).flatten()
                df_processed.loc[group_df.index, f'{metric}_z'] = z_scores

        metric_cols = [col for col in df_processed.columns if '_90' in col or '_ratio' in col or 'length' in col]
        pct_cols = [col for col in df_processed.columns if '_pct' in col]
        z_cols = [col for col in df_processed.columns if '_z' in col]
        cols_to_clean = list(set(metric_cols + pct_cols + z_cols))
        df_processed[cols_to_clean] = df_processed[cols_to_clean].fillna(0)

        if 'season_name' in df_processed.columns:
            df_processed['canonical_season'] = df_processed['season_name'].apply(get_canonical_season)

        return df_processed

    # --- 5. ANALYSIS & REPORTING FUNCTIONS ---

    def find_player_by_name(df, player_name):
        if not player_name: return None, None
        exact_matches = df[df['player_name'].str.lower() == player_name.lower()]
        if not exact_matches.empty: return exact_matches.iloc[0].copy(), None

        partial_matches = df[df['player_name'].str.lower().str.contains(player_name.lower(), na=False)]
        if not partial_matches.empty:
            suggestions = partial_matches[['player_name', 'team_name']].head(5).to_dict('records')
            return None, suggestions
        return None, None

    def detect_player_archetype(target_player, archetypes):
        archetype_scores = {}
        for name, config in archetypes.items():
            metrics = [f"{m}_pct" for m in config['identity_metrics']]
            valid_metrics = [m for m in metrics if m in target_player.index and pd.notna(target_player[m])]
            score = target_player[valid_metrics].mean() if valid_metrics else 0
            archetype_scores[name] = score

        best_archetype = max(archetype_scores, key=archetype_scores.get) if archetype_scores else None
        return best_archetype, pd.DataFrame(archetype_scores.items(), columns=['Archetype', 'Affinity Score']).sort_values(by='Affinity Score', ascending=False)


def find_matches(target_player, pool_df, archetype_config, season_df=None, search_mode="similar", min_minutes=600, top_n=100):
    """Two-tier similarity search.

    Returns candidates in two tiers:
      - True Clones: tight agreement on defining traits + high similarity + adequate coverage
      - Next Best Fits: nearest neighbors filling the remaining slots

    Always attempts to return enough rows to populate a Top 10 (when the pool has them),
    while keeping the 'true clone' label honest.

    Notes:
      - Uses UNION of identity metrics across archetypes for the target's position_group (profile stability).
      - Uses robust Mahalanobis distance (LedoitWolf) when sample size allows, with safe fallbacks.
      - Never treats missing metrics as 'average' for clone qualification; coverage is penalized.
    """
    if target_player is None or pool_df is None or pool_df.empty:
        return pd.DataFrame()

    df = pool_df.copy()

    # --- Pool filtering ---
    if "minutes" in df.columns:
        df = df[df["minutes"].fillna(0) >= float(min_minutes)]

    if "player_id" in df.columns and "player_id" in target_player.index:
        df = df[df["player_id"] != target_player["player_id"]]

    tgt_group = target_player.get("position_group", None)
    if tgt_group is not None and "position_group" in df.columns:
        df = df[df["position_group"] == tgt_group]

    if df.empty:
        return pd.DataFrame()

    # --- Metric space (UNION across archetypes in this position group) ---
    try:
        grp_cfg = POSITIONAL_CONFIGS.get(tgt_group, {})
        archs = grp_cfg.get("archetypes", {})
        union_metrics = set()
        for _, cfg in archs.items():
            for m in cfg.get("identity_metrics", []):
                union_metrics.add(m)
        if not union_metrics:
            union_metrics = set(archetype_config.get("identity_metrics", []))
        z_cols = [f"{m}_z" for m in sorted(union_metrics)]
    except Exception:
        z_cols = [f"{m}_z" for m in archetype_config.get("identity_metrics", [])]

    z_cols = [c for c in z_cols if c in df.columns and c in target_player.index]
    if not z_cols:
        return pd.DataFrame()

    X = df[z_cols].apply(pd.to_numeric, errors="coerce")
    t = pd.to_numeric(target_player[z_cols], errors="coerce")

    # --- Coverage (shared observed dimensions) ---
    cand_cov = X.notna().mean(axis=1).clip(0, 1)
    tgt_cov = float(t.notna().mean()) if len(t) else 0.0
    combined_cov = (cand_cov * tgt_cov) ** 0.5

    # --- Defining traits (top-K spikes in abs z) ---
    t_filled = t.fillna(0.0)
    abs_z = t_filled.abs()

    # Parameters (sane defaults)
    clone_def_k = int(archetype_config.get("clone_def_k", 6))
    clone_def_k = max(4, min(10, clone_def_k, len(z_cols)))

    clone_def_tol = float(archetype_config.get("clone_def_tol_z", 0.6))   # ± z window for defining metrics
    clone_match_need = int(archetype_config.get("clone_match_need", clone_def_k - 1))  # e.g., 5/6
    clone_match_need = max(2, min(clone_def_k, clone_match_need))

    clone_sim_floor = float(archetype_config.get("clone_sim_floor", 60.0))
    clone_cov_floor = float(archetype_config.get("clone_cov_floor", 0.70))

    defining_cols = abs_z.sort_values(ascending=False).head(clone_def_k).index.tolist()

    # Count defining agreements
    def_diffs = (X[defining_cols].sub(t_filled[defining_cols], axis=1)).abs()
    def_match_count = (def_diffs <= clone_def_tol).sum(axis=1).astype(int)

    # A soft defining match score for ranking ties
    def_mean = def_diffs.mean(axis=1)
    defining_match_score = np.exp(-def_mean)  # 1 is best

    # --- Robust distance (Mahalanobis when possible) ---
    X_complete = X.dropna(axis=0, how="any")
    n_complete = len(X_complete)
    n_feat = len(z_cols)
    ridge = 1e-3

    if n_complete >= max(30, 2 * n_feat):
        try:
            lw = LedoitWolf()
            lw.fit(X_complete.to_numpy(dtype=float))
            cov = lw.covariance_
        except Exception:
            cov = np.cov(X_complete.to_numpy(dtype=float), rowvar=False)
            cov = cov + ridge * np.eye(n_feat, dtype=float)
    elif n_complete >= max(10, n_feat + 5):
        cov = np.cov(X_complete.to_numpy(dtype=float), rowvar=False)
        cov = cov + (2 * ridge) * np.eye(n_feat, dtype=float)
    else:
        # very small sample: diagonalized correlation fallback
        corr = X_complete.corr().fillna(0.0).to_numpy()
        stds = X_complete.std().fillna(1.0).to_numpy()
        cov = np.outer(stds, stds) * corr
        cov = cov + (3 * ridge) * np.eye(n_feat, dtype=float)

    try:
        VI = np.linalg.pinv(cov)
    except Exception:
        VI = np.eye(n_feat, dtype=float)

    X_f = X.fillna(0.0).to_numpy(dtype=float)
    t_vec = t_filled.to_numpy(dtype=float).reshape(1, -1)
    diffs = X_f - t_vec

    try:
        mahal_sq = np.einsum("ij,jk,ik->i", diffs, VI, diffs)
        mahal_sq = np.maximum(mahal_sq, 0.0)
        dists = np.sqrt(mahal_sq)
    except Exception:
        dists = np.linalg.norm(diffs, axis=1)

    # --- Similarity score (0..100) ---
    base_sim = 100.0 * np.exp(-0.50 * dists)
    sim = base_sim * (combined_cov.to_numpy(dtype=float) ** 0.85)
    sim = sim * (0.85 + 0.15 * defining_match_score.to_numpy(dtype=float))
    sim = np.clip(sim, 0.0, 100.0)

    out = df.copy()
    out["similarity_score"] = sim
    out["_coverage"] = combined_cov
    out["_defining_match_score"] = defining_match_score
    out["_defining_match_count"] = def_match_count
    out["_defining_k"] = clone_def_k
    out["_defining_tol_z"] = clone_def_tol
    out["_mahal_dist"] = dists

    # --- Two-tier labeling ---
    is_clone = (
        (out["_defining_match_count"] >= clone_match_need) &
        (out["similarity_score"] >= clone_sim_floor) &
        (out["_coverage"] >= clone_cov_floor)
    )

    out["match_tier"] = np.where(is_clone, "True Clone", "Next Best Fit")

    # Fail reasons for transparency (helps you tune profile traits without guessing)
    fail = []
    for i in range(len(out)):
        if is_clone.iloc[i]:
            fail.append("")
            continue
        reasons = []
        if out["_defining_match_count"].iloc[i] < clone_match_need:
            reasons.append(f"defining {int(out['_defining_match_count'].iloc[i])}/{clone_def_k}")
        if out["similarity_score"].iloc[i] < clone_sim_floor:
            reasons.append("similarity floor")
        if out["_coverage"].iloc[i] < clone_cov_floor:
            reasons.append("low coverage")
        fail.append(", ".join(reasons))
    out["_fail_reason"] = fail

    # --- Ranking / output ---
    true_clones = out[out["match_tier"] == "True Clone"].sort_values(
        ["similarity_score", "_defining_match_count", "_coverage"],
        ascending=[False, False, False]
    )

    neighbors = out[out["match_tier"] == "Next Best Fit"].sort_values(
        ["similarity_score", "_defining_match_count", "_coverage"],
        ascending=[False, False, False]
    )

    # Always build a Top 10 view (or more if requested)
    want = int(max(10, top_n))
    if len(true_clones) >= want:
        res = true_clones.head(want)
    else:
        need = max(0, want - len(true_clones))
        res = pd.concat([true_clones, neighbors.head(need)], ignore_index=True)

    # Preserve upgrade mode behavior (if UI uses it)
    if search_mode == "upgrade":
        # For upgrade mode, we *still* keep clone-first ordering, but surface upgrade_score for sorting within tiers.
        pct_cols = []
        try:
            grp_cfg = POSITIONAL_CONFIGS.get(tgt_group, {})
            archs = grp_cfg.get("archetypes", {})
            union_metrics = set()
            for _, cfg in archs.items():
                for m in cfg.get("identity_metrics", []):
                    union_metrics.add(m)
            if not union_metrics:
                union_metrics = set(archetype_config.get("identity_metrics", []))
            pct_cols = [f"{m}_pct" for m in sorted(union_metrics)]
        except Exception:
            pct_cols = [f"{m}_pct" for m in archetype_config.get("identity_metrics", [])]

        pct_cols = [c for c in pct_cols if c in res.columns]
        if pct_cols:
            res["upgrade_score"] = res[pct_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            res = res.sort_values(
                ["match_tier", "upgrade_score", "similarity_score"],
                ascending=[True, False, False]
            )

    return res


    def _radar_angles_labels(metrics_dict):
        labels = list(metrics_dict.values())
        metrics = list(metrics_dict.keys())
        return metrics, labels

    def _player_percentiles_for_metrics(player_series, metrics):
        return [float(player_series.get(f"{m}_pct", 0.0)) for m in metrics]

    def create_plotly_radar(players_data, radar_config, bg_color="#111111"):
        """Generates a Plotly Figure for a radar chart with multiple players."""
        metrics_dict = radar_config['metrics']
        group_name = radar_config['name']
        metrics, labels = _radar_angles_labels(metrics_dict)

        palette = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#FFC0CB']
        fallback_palette = ["#FFFF00", "#00FFFF", "#800080", "#FFD700"]
        full_palette = palette + fallback_palette

        fig = go.Figure()

        for i, player_series in enumerate(players_data):
            player_name = player_series.get('player_name', 'Unknown')
            season_name = player_series.get('season_name', 'Unknown')
            label = f"{player_name} ({season_name})"
            color = full_palette[i % len(full_palette)]

            rgb_color = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
            rgba_fillcolor = f'rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.2)'

            percentile_values = _player_percentiles_for_metrics(player_series, metrics)

            trace = go.Scatterpolar(
                r=percentile_values + [percentile_values[0]],
                theta=labels + [labels[0]],
                mode="lines+markers+text",
                name=label,
                line=dict(width=2, color=color),
                marker=dict(size=5, color=color),
                text=[f"{int(round(v))}" for v in percentile_values] + [f"{int(round(percentile_values[0]))}"],
                textfont=dict(size=11, color="#ffffff" if len(players_data) == 1 else "rgba(0,0,0,0)"),
                textposition="top center",
                hovertemplate="%{theta}<br>%{r:.0f}th percentile<extra>" + label + "</extra>",
                fill="toself",
                fillcolor=rgba_fillcolor,
                opacity=0.8,
                legendgroup=label,
                hoveron="points+fills",
            )
            fig.add_trace(trace)

        fig.update_layout(
            title=dict(
                text=group_name, x=0.5, xanchor='center',
                y=0.95, yanchor='top', font=dict(size=18, color="white"),
                pad=dict(t=24, b=4, l=4, r=4)
            ),
            showlegend=True,
            legend=dict(
                orientation="h", x=0.5, xanchor="center",
                y=-0.15, yanchor="top", font=dict(size=11, color="white"),
                itemsizing="trace"
            ),
            polar=dict(
                bgcolor=bg_color,
                radialaxis=dict(range=[0, 100], showline=False, showticklabels=True, tickfont=dict(color="white", size=10),
                                 gridcolor="rgba(255,255,255,0.15)", tickangle=0),
                angularaxis=dict(
                    tickvals=list(range(len(labels))), ticktext=labels,
                    tickfont=dict(size=11, color="white"),
                    gridcolor="rgba(255,255,255,0.1)"
                )
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            margin=dict(t=80, b=90, l=40, r=40),
            hovermode="closest"
        )
        fig.update_layout(height=520)
        return fig, metrics

    def render_plotly_with_legend_hover(fig, metrics, height=520, player_names=None):
        """Adds a checkbox to highlight a player and a button to view the radar in a fullscreen dialog."""
        unique_key = fig.layout.title.text.replace(" ", "_").replace(":", "").lower()

        if st.button("👁️ View Fullscreen", key=f"fullscreen_{unique_key}"):
            with st.dialog(f"Fullscreen Radar: {fig.layout.title.text}"):
                dialog_fig = go.Figure(fig)
                dialog_fig.update_layout(height=700, title_font_size=24, legend_font_size=14)
                st.plotly_chart(dialog_fig, use_container_width=True)

        highlight = st.checkbox("Highlight a player on this radar", key=f"highlight_{unique_key}")
        selected_player = None
        if highlight and player_names:
            selected_player = st.selectbox("Select player", player_names, key=f"player_select_{unique_key}", index=None, placeholder="Select a player to highlight")

        display_fig = go.Figure(fig)
        if selected_player:
            for i, trace in enumerate(display_fig.data):
                player_legend_name = trace.name
                if selected_player in player_legend_name:
                    display_fig.data[i].opacity = 1.0
                    display_fig.data[i].textfont.color = "#ffffff"
                else:
                    display_fig.data[i].opacity = 0.2
                    display_fig.data[i].textfont.color = "rgba(0,0,0,0)"

        st.plotly_chart(display_fig, use_container_width=True, height=height)


    def project_external_to_internal_distributions(internal_df: pd.DataFrame, external_df: pd.DataFrame):
        """Projects external rows (e.g., open data) into the internal z/pct space per position_group.

        - Uses internal distribution per position_group for each metric.
        - Leaves missing metrics as NaN (critical for clone distance on shared dims).
        - Applies the same negative-stat inversion used in process_data.
        """
        if external_df is None or external_df.empty:
            return pd.DataFrame()

        ext = external_df.copy()
        # derive position_group with the same mapping
        def get_position_group(primary_position):
            for group, config in POSITIONAL_CONFIGS.items():
                if primary_position in config['positions']:
                    return group
            return None
        ext['position_group'] = ext.get('primary_position', pd.Series([None]*len(ext))).apply(get_position_group)

        negative_stats = ['turnovers_90', 'dispossessions_90', 'dribbled_past_90', 'fouls_90']

        for metric in ALL_METRICS_TO_PERCENTILE:
            if metric not in ext.columns:
                ext[metric] = np.nan

            ext[f'{metric}_pct'] = np.nan
            ext[f'{metric}_z'] = np.nan

            for group, ext_g in ext.groupby('position_group', dropna=False):
                if group is None:
                    continue
                int_g = internal_df[internal_df['position_group'] == group]
                if int_g.empty or metric not in int_g.columns:
                    continue

                base = int_g[metric].replace([np.inf, -np.inf], np.nan).dropna()
                if base.empty:
                    continue

                mean = base.mean()
                std = base.std(ddof=0) if base.std(ddof=0) > 1e-9 else np.nan

                sorted_vals = np.sort(base.values)

                for idx in ext_g.index:
                    x = ext.at[idx, metric]
                    if pd.isna(x):
                        continue

                    # percentile within internal distribution
                    # rank = proportion of internal <= x
                    r = np.searchsorted(sorted_vals, x, side='right') / len(sorted_vals)

                    if metric in negative_stats:
                        pct = (1 - r) * 100
                    else:
                        pct = r * 100

                    ext.at[idx, f'{metric}_pct'] = pct

                    if std is not np.nan and pd.notna(std) and std > 0 and pd.notna(x):
                        z = (x - mean) / std
                        ext.at[idx, f'{metric}_z'] = float(z)

        return ext

    def get_season_start_year(season_str):
        """
        Intelligently extracts the starting year from a season string (e.g., '2024' from '2024/2025' or '2025' from '2025').
        This ensures correct chronological sorting.
        """
        try:
            if isinstance(season_str, str) and '/' in season_str:
                return int(season_str.split('/')[0])
            else:
                return int(season_str)
        except (ValueError, TypeError):
            return 0

    # --- 7. STREAMLIT APP LAYOUT ---
    st.title("⚽ Advanced Multi-Position Player Analysis v12.0")

    processed_data = None
    try:
        with st.spinner("Loading and processing data for all leagues... This may take a minute."):
            raw_data = get_all_leagues_data((USERNAME, PASSWORD))
            if raw_data is not None:
                processed_data = process_data(raw_data)
            else:
                st.error("Failed to load data. Please check credentials and connection.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure StatsBomb credentials are configured in Codespaces secrets.")

    scouting_tab, comparison_tab = st.tabs(["Scouting", "Direct Comparison"])

    def create_player_filter_ui(data, key_prefix, pos_filter=None):
        leagues = sorted(data['league_name'].dropna().unique())

        selected_league = st.selectbox("League", leagues, key=f"{key_prefix}_league", index=None, placeholder="Choose a league")

        if selected_league:
            league_df = data[data['league_name'] == selected_league]
            seasons = sorted(league_df['season_name'].unique(), key=get_season_start_year, reverse=True)
            selected_season = st.selectbox("Season", seasons, key=f"{key_prefix}_season", index=None, placeholder="Choose a season")

            if selected_season:
                season_df = league_df[league_df['season_name'] == selected_season]

                if pos_filter:
                    config = POSITIONAL_CONFIGS.get(pos_filter, {})
                    valid_positions = config.get('positions', [])
                    season_df_filtered = season_df[season_df['primary_position'].isin(valid_positions)]

                    if season_df_filtered.empty and not season_df.empty:
                        available_pos = sorted(season_df['primary_position'].unique())
                        st.warning(f"No players found for '{pos_filter}'. Available positions in this selection: {available_pos}")
                        return None
                    season_df = season_df_filtered

                teams = ["All Teams"] + sorted(season_df['team_name'].unique())
                selected_team = st.selectbox("Team", teams, key=f"{key_prefix}_team")

                if selected_team:
                    if selected_team != "All Teams":
                        player_pool = season_df[season_df['team_name'] == selected_team]
                    else:
                        player_pool = season_df

                    if player_pool.empty:
                        st.warning(f"No players found for the selected filters.")
                        return None

                    player_pool_display = player_pool.copy()
                    player_pool_display['age_str'] = player_pool_display['age'].apply(lambda x: str(int(x)) if pd.notna(x) else 'N/A')
                    player_pool_display['display_name'] = player_pool_display['player_name'] + " (" + player_pool_display['age_str'] + ", " + player_pool_display['primary_position'].fillna('N/A') + ")"

                    players = sorted(player_pool_display['display_name'].unique())
                    selected_display_name = st.selectbox("Player", players, key=f"{key_prefix}_player", index=None, placeholder="Choose a player")

                    if selected_display_name:
                        player_instance_df = player_pool_display[player_pool_display['display_name'] == selected_display_name]
                        if not player_instance_df.empty:
                            original_index = player_instance_df.index[0]
                            return data.loc[original_index]
        return None

    with scouting_tab:
        if processed_data is not None:
            st.sidebar.header("🔍 Scouting Controls")
            pos_options = list(POSITIONAL_CONFIGS.keys())
            selected_pos = st.sidebar.selectbox("1. Select Position", pos_options, key="scout_pos")
            filter_by_pos = st.sidebar.checkbox("Filter dropdowns by position group", value=True, key="pos_filter_toggle")

            league_filter_options = ["All Leagues", "Domestic Leagues", "Scottish Leagues"]
            selected_league_filter = st.sidebar.selectbox("League Filter", league_filter_options, key="league_filter")

            st.sidebar.subheader("Select Target Player")
            min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 3000, 600, 100)
            age_range = st.sidebar.slider("Age Range", 16, 40, (16, 40), key="age_range")
            pos_filter_arg = selected_pos if filter_by_pos else None
            target_player = create_player_filter_ui(processed_data, key_prefix="scout", pos_filter=pos_filter_arg)

            search_mode = st.sidebar.radio("Search Mode", ('Find Similar Players', 'Find Potential Upgrades'), key='scout_mode')
            search_mode_logic = 'upgrade' if search_mode == 'Find Potential Upgrades' else 'similar'

            search_scope = st.sidebar.selectbox(
                "Search Scope",
                ('Last Season Only', 'Last 2 Seasons', 'All Historical Data'),
                key='scout_scope'
            )

            if st.sidebar.button("Analyze Player", type="primary", key="scout_analyze") and target_player is not None:
                st.session_state.analysis_run = True
                st.session_state.target_player = target_player
                st.session_state.radar_players = []

                config = POSITIONAL_CONFIGS[selected_pos]
                st.session_state.analysis_pos = selected_pos
                archetypes = config["archetypes"]

                target_pos_group = target_player['position_group']
                if pd.isna(target_pos_group):
                    st.error("Target player position group could not be determined. Cannot find matches.")
                    st.session_state.matches = pd.DataFrame()
                else:
                    position_pool = processed_data[processed_data['position_group'] == target_pos_group]

                    detected_archetype, dna_df = detect_player_archetype(target_player, archetypes)
                    st.session_state.detected_archetype = detected_archetype
                    st.session_state.dna_df = dna_df

                    if detected_archetype:
                        archetype_config = archetypes[detected_archetype]

                        canonical_seasons = sorted(position_pool['canonical_season'].unique(), reverse=True)

                        seasons_to_search = canonical_seasons
                        if search_scope == 'Last Season Only':
                            seasons_to_search = canonical_seasons[:1]
                        elif search_scope == 'Last 2 Seasons':
                            seasons_to_search = canonical_seasons[:2]

                        search_pool = position_pool[position_pool['canonical_season'].isin(seasons_to_search)]

                        # Apply league filter
                        if selected_league_filter == "Domestic Leagues" and 'competition_id' in search_pool.columns:
                            search_pool = search_pool[search_pool['competition_id'].isin(DOMESTIC_LEAGUE_IDS)]
                        elif selected_league_filter == "Scottish Leagues" and 'competition_id' in search_pool.columns:
                            search_pool = search_pool[search_pool['competition_id'].isin(SCOTTISH_LEAGUE_IDS)]

                        # Apply age filter
                        unknown_age_count = 0
                        if 'age' in search_pool.columns:
                            unknown_age_count = search_pool['age'].isna().sum()
                            known_ages = search_pool[search_pool['age'].notna()]
                            filtered_known = known_ages[(known_ages['age'] >= age_range[0]) & (known_ages['age'] <= age_range[1])]
                            search_pool = pd.concat([filtered_known, search_pool[search_pool['age'].isna()]], ignore_index=True)

                        st.session_state.unknown_age_count = unknown_age_count

                        matches = find_matches(
                            target_player,
                            search_pool,
                            archetype_config,
                            search_mode_logic,
                            min_minutes
                        )
                        st.session_state.matches = matches
                    else:
                        st.session_state.matches = pd.DataFrame()

                st.rerun()

            if st.session_state.analysis_run and 'target_player' in st.session_state and st.session_state.target_player is not None:
                tp = st.session_state.target_player
                selected_pos = tp['position_group'] if pd.notna(tp['position_group']) else selected_pos

                st.header(f"Analysis: {tp['player_name']} ({tp['primary_position']} | {tp['season_name']})")

                if st.session_state.detected_archetype:
                    st.subheader(f"Detected Archetype: {st.session_state.detected_archetype}")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(st.session_state.dna_df.reset_index(drop=True), hide_index=True)
                    with col2:
                        analysis_pos = st.session_state.get("analysis_pos", selected_pos)
                        pos_cfg = POSITIONAL_CONFIGS.get(analysis_pos, {})
                        archetypes_cfg = pos_cfg.get("archetypes", {})
                        arch_cfg = archetypes_cfg.get(st.session_state.detected_archetype)
                        desc = arch_cfg.get("description") if arch_cfg else "Description not found for this archetype under the selected position set."
                        st.write(f"**Description**: {desc}")
                        st.subheader(f"Top 10 Matches ({search_mode})")
                        if st.session_state.matches is not None and not st.session_state.matches.empty:
                            if st.session_state.get('unknown_age_count', 0) > 0:
                                st.caption(f"Including {st.session_state.unknown_age_count} players with unknown ages.")

                            display_cols = ['player_name', 'age', 'primary_position', 'team_name', 'league_name', 'season_name']
                            score_col = 'upgrade_score' if search_mode_logic == 'upgrade' else 'similarity_score'
                            display_cols.insert(1, score_col)

                            matches_df = st.session_state.matches.copy()
                            matches_df[score_col] = matches_df[score_col].round(1)

                            # --- Split into tiers for clone-style similarity
                            if search_mode_logic != 'upgrade' and 'match_tier' in matches_df.columns:
                                clones = matches_df[matches_df['match_tier'] == 'Perfect Clone'].head(10)
                                next_best = matches_df[matches_df['match_tier'] == 'Next Best'].head(10)

                                if not clones.empty:
                                    st.markdown("### Perfect Clones")
                                    st.dataframe(
                                        clones[display_cols].rename(columns=lambda c: c.replace('_', ' ').title()),
                                        hide_index=True,
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("No 'Perfect Clone' matches under the current pool/minutes. Showing next-best fits below.")

                                st.markdown("### Next Best Fits")
                                st.dataframe(
                                    next_best[display_cols].rename(columns=lambda c: c.replace('_', ' ').title()),
                                    hide_index=True,
                                    use_container_width=True,
                                )

                                subset_keys = [c for c in ['player_id', 'season_id'] if c in matches_df.columns]
                                btn_df = pd.concat([clones, next_best], axis=0)
                                if subset_keys:
                                    btn_df = btn_df.drop_duplicates(subset=subset_keys)
                                btn_df = btn_df.head(10)
                            else:
                                # upgrade or legacy
                                btn_df = matches_df.head(10)
                                st.dataframe(
                                    btn_df[display_cols].rename(columns=lambda c: c.replace('_', ' ').title()),
                                    hide_index=True,
                                    use_container_width=True,
                                )

                            st.subheader("Add Players to Radar Comparison")
                            for _, row in btn_df.iterrows():
                                btn_key = f"add_{row.get('player_id','x')}_{row.get('season_id','y')}"
                                age_str = str(int(row['age'])) if pd.notna(row.get('age')) else 'N/A'
                                team_str = row.get('team_name', 'Unknown Team')
                                button_label = f"Add {row['player_name']} ({age_str}, {team_str})"
                                if st.button(button_label, key=btn_key):
                                    if not any(
                                        p['player_id'] == row['player_id'] and p['season_id'] == row['season_id']
                                        for p in st.session_state.radar_players
                                    ):
                                        st.session_state.radar_players.append(row)
                                        st.rerun()
                        else:
                            st.warning("No matching players found with the current filters.")

                if st.session_state.radar_players:
                    st.subheader("Players on Radar")
                    num_players_on_radar = len(st.session_state.radar_players)
                    radar_cols = st.columns(num_players_on_radar or 1)
                    for i in range(num_players_on_radar):
                        with radar_cols[i]:
                            player_data = st.session_state.radar_players[i]
                            age_str = str(int(player_data['age'])) if pd.notna(player_data['age']) else 'N/A'
                            st.markdown(f"**{player_data['player_name']}** ({age_str})")
                            st.markdown(f"{player_data['primary_position']} | {player_data['team_name']}")
                            st.markdown(f"`{player_data['league_name']} - {player_data['season_name']}`")
                            if st.button("❌ Remove", key=f"remove_scout_{i}"):
                                st.session_state.radar_players.pop(i)
                                st.rerun()

                st.subheader("Player Radars")
                players_to_show = [st.session_state.target_player] + st.session_state.radar_players

                if selected_pos and selected_pos in POSITIONAL_CONFIGS:
                    radars_to_show = POSITIONAL_CONFIGS[selected_pos]['radars']
                    num_radars = len(radars_to_show)
                    cols = st.columns(3)
                    radar_items = list(radars_to_show.items())

                    for i in range(num_radars):
                        with cols[i % 3]:
                            radar_key, radar_config = radar_items[i]
                            player_names = [p['player_name'] for p in players_to_show]
                            fig, metrics = create_plotly_radar(players_to_show, radar_config)
                            render_plotly_with_legend_hover(fig, metrics, height=520, player_names=player_names)
                else:
                     st.warning("Select a player and run analysis to see radar charts.")

            else:
                st.info("Select a position and target player from the sidebar, then click 'Analyze Player' to begin.")

        else:
            st.error("Data could not be loaded. Please check your credentials in the script and your internet connection.")

    with comparison_tab:
        st.header("Multi-Player Direct Comparison")

        if processed_data is not None:
            def player_filter_ui_comp(data, key_prefix):
                state = st.session_state.comp_selections

                leagues = sorted(data['league_name'].dropna().unique())

                league_idx = leagues.index(state['league']) if state['league'] in leagues else None
                selected_league = st.selectbox("League", leagues, key=f"{key_prefix}_league", index=league_idx, placeholder="Choose a league")

                if selected_league and selected_league != state.get('league'):
                    state['league'] = selected_league
                    state['season'] = None
                    state['team'] = None
                    state['player'] = None
                    st.rerun()

                if state.get('league'):
                    league_df = data[data['league_name'] == state['league']]
                    seasons = sorted(league_df['season_name'].unique(), key=get_season_start_year, reverse=True)
                    season_idx = seasons.index(state['season']) if state.get('season') in seasons else None
                    selected_season = st.selectbox("Season", seasons, key=f"{key_prefix}_season", index=season_idx, placeholder="Choose a season")

                    if selected_season and selected_season != state.get('season'):
                        state['season'] = selected_season
                        state['team'] = None
                        state['player'] = None
                        st.rerun()

                if state.get('season'):
                    season_df = data[(data['league_name'] == state['league']) & (data['season_name'] == state['season'])]
                    teams = ["All Teams"] + sorted(season_df['team_name'].unique())
                    team_idx = teams.index(state['team']) if state.get('team') in teams else 0
                    selected_team = st.selectbox("Team", teams, key=f"{key_prefix}_team", index=team_idx)

                    if selected_team and selected_team != state.get('team'):
                        state['team'] = selected_team
                        state['player'] = None
                        st.rerun()

                if state.get('team'):
                    if state['team'] != "All Teams":
                        player_pool = data[
                            (data['league_name'] == state['league']) & 
                            (data['season_name'] == state['season']) & 
                            (data['team_name'] == state['team'])
                        ]
                    else:
                        player_pool = data[
                            (data['league_name'] == state['league']) & 
                            (data['season_name'] == state['season'])
                        ]

                    if not player_pool.empty:
                        player_pool_display = player_pool.copy()
                        player_pool_display['age_str'] = player_pool_display['age'].apply(lambda x: str(int(x)) if pd.notna(x) else 'N/A')
                        player_pool_display['display_name'] = (
                            player_pool_display['player_name'] + 
                            " (" + player_pool_display['age_str'] + 
                            ", " + player_pool_display['primary_position'].fillna('N/A') + ")"
                        )

                        players = sorted(player_pool_display['display_name'].unique())
                        player_idx = players.index(state['player']) if state.get('player') in players else None

                        selected_display_name = st.selectbox("Player", players, key=f"{key_prefix}_player_detailed", index=player_idx, placeholder="Choose a player to add")

                        if selected_display_name and selected_display_name != state.get('player'):
                            state['player'] = selected_display_name
                            st.rerun()

                        if state.get('player'):
                            player_instance_df = player_pool_display[player_pool_display['display_name'] == state['player']]
                            if not player_instance_df.empty:
                                return data.loc[player_instance_df.index[0]]
                return None

            with st.container(border=True):
                st.subheader("Add a Player to Comparison")
                player_instance = player_filter_ui_comp(processed_data, key_prefix="comp")

                if st.button("Add Player to Comparison", type="primary"):
                    if player_instance is not None:
                        player_id = f"{player_instance['player_id']}_{player_instance['season_id']}"
                        if not any(f"{p['player_id']}_{p['season_id']}" == player_id for p in st.session_state.comparison_players):
                            st.session_state.comparison_players.append(player_instance)
                            st.rerun()
                        else:
                            st.warning("This player and season is already in the comparison.")
                    else:
                        st.warning("Please select a valid player from all dropdowns.")

            st.divider()

            st.subheader("Current Comparison")
            if not st.session_state.comparison_players:
                st.info("Add one or more players using the selection box above to start a comparison.")
            else:
                num_comp_players = len(st.session_state.comparison_players)
                player_cols = st.columns(num_comp_players or 1)
                for i in range(num_comp_players):
                    with player_cols[i]:
                        player_data = st.session_state.comparison_players[i]
                        age_str = str(int(player_data['age'])) if pd.notna(player_data['age']) else 'N/A'
                        st.markdown(f"**{player_data['player_name']}** ({age_str})")
                        st.markdown(f"{player_data['primary_position']} | *{player_data['team_name']}*")
                        st.markdown(f"`{player_data['league_name']} - {player_data['season_name']}`")
                        if st.button("❌ Remove", key=f"remove_comp_{i}"):
                            st.session_state.comparison_players.pop(i)
                            st.rerun()

            st.divider()

            if st.session_state.comparison_players:
                st.subheader("Radar Chart Comparison")

                pos_groups = [p['position_group'] for p in st.session_state.comparison_players if pd.notna(p['position_group'])]
                if pos_groups:
                    default_pos = max(set(pos_groups), key=pos_groups.count)
                else:
                    default_pos = "Striker"

                radar_pos_options = list(POSITIONAL_CONFIGS.keys())
                default_index = radar_pos_options.index(default_pos) if default_pos in radar_pos_options else 0

                selected_radar_pos = st.selectbox("Select Radar Set to Use for Comparison", radar_pos_options, index=default_index)

                if selected_radar_pos:
                    radars_to_show = POSITIONAL_CONFIGS[selected_radar_pos]['radars']

                    num_radars = len(radars_to_show)
                    cols = st.columns(3) 
                    radar_items = list(radars_to_show.items())

                    for i in range(num_radars):
                        with cols[i % 3]:
                            radar_key, radar_config = radar_items[i]
                            player_names_for_hover = [p['player_name'] for p in st.session_state.comparison_players]
                            fig, metrics = create_plotly_radar(st.session_state.comparison_players, radar_config)
                            render_plotly_with_legend_hover(fig, metrics, height=520, player_names=player_names_for_hover)
        else:
            st.error("Data could not be loaded. Please check your credentials in the script.")

if __name__ == "__main__":
    main()
