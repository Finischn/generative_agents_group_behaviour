import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import imageio.v2 as imageio
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib.patches import Patch
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, FancyArrowPatch
import seaborn as sns
import tempfile
#import imageio


matplotlib.rcParams.update({
    "font.family": "serif",   # Matches LaTeX's default font
    "font.size": 12,          # Matches LaTeX 12pt
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
})

# Storage directory 
storage_dir = "../environment/frontend_server/storage"
# Define groups for race and village
groups_race = {
        'Isabella': 'Black',
        'Ayesha': 'Black',
        'Carlos': 'Black',
        'Tamara': 'Black',
        'Wolfgang': 'Black',
        'Sam': 'White',
        'Adam': 'White',
        'Eddy': 'White',
        'Klaus': 'White',
        'Tom': 'White'
    }

groups_village = {
        'Isabella': 'newcomer to the village',
        'Ayesha': 'newcomer to the village',
        'Carlos': 'newcomer to the village',
        'Tamara': 'newcomer to the village',
        'Wolfgang': 'newcomer to the village',
        'Sam': 'longterm resident of the village',
        'Adam': 'longterm resident of the village',
        'Eddy': 'longterm resident of the village',
        'Klaus': 'longterm resident of the village',
        'Tom': 'longterm resident of the village'
    }
    

def load_simulation_data(simulations, target_dir):
    """
    Load JSON data from 'decide_talk.json' or conversations.json files in each simulation folder and 
    return a dictionary of DataFrames.

    Parameters:
    simulations (list): List of simulation folder names.
    target_dir (str): name of target directory.

    Returns:
    dict: A dictionary where keys are folder names and values are DataFrames containing simulation data.
    """
    simulation_dataframes = {}

    for folder in simulations:
        folder_path = os.path.join(storage_dir, folder)
        if target_dir == 'decide_talk':
            decide_talk_path = os.path.join(folder_path, 'decide_talk', 'decide_talk.json')
        else:
            decide_talk_path = os.path.join(folder_path, 'conversations', 'conversations.json')
        
        if os.path.exists(decide_talk_path):
            print(f"Opening file in {folder}")
            
            with open(decide_talk_path, 'r') as json_file:
                data = json.load(json_file)
                
                flat_data = []
                for conversation in data:
                    conversation.update({"simulation": folder})
                    flat_data.append(conversation)
                
                simulation_dataframes[folder] = pd.DataFrame(flat_data)
                print(f"DataFrame created for simulation: {folder}")
        else:
            print(f"file not found in {folder} at path: {decide_talk_path}")

    return simulation_dataframes



def analyze_and_plot_conversations(conv_summaries, ordered_agents, groups_race):

    """
    Compute termination matrices for each simulation and plot heatmaps.



    Parameters:
    - conv_summaries: Dictionary of conversation summaries per simulation
    - ordered_agents: List of agents sorted by group
    - groups_race: Dictionary mapping agents to racial groups
    """
    termination_matrices = {}
    for sim_name, conv_summary in conv_summaries.items():

        # Compute the termination matrix
        termination_matrix = compute_termination_matrix(conv_summary, ordered_agents)



        # Find the index where the second group starts (for boundary lines)
        split_index = next(i for i, agent in enumerate(ordered_agents) if groups_race[agent] == "White")



        # Plot heatmap
        plot_heatmap_convs(sim_name, termination_matrix, split_index)
        termination_matrices[sim_name] = termination_matrix
    return termination_matrices

def plot_heatmap_convs(sim_name, matrix, split_index):
    """
    Plot a heatmap for a given termination matrix with boundary lines.



    Parameters:
    - sim_name: Name of the simulation
    - matrix: The termination matrix
    - split_index: Index where the second group starts (to draw a separation line)
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Oranges", linewidths=0.5)



    # Draw boundary lines between groups
    plt.axvline(x=split_index, color='black', linewidth=2)
    plt.axhline(y=split_index, color='black', linewidth=2)



    plt.title(f"{sim_name} - Termination Matrix")
    plt.xlabel("Agent Ending Conversation")
    plt.ylabel("Agent Pair Interaction")
    plt.show()


def create_interaction_gif(simulation_dataframes, simulation_names):
    all_agents = sorted(set().union(*[set(df['init_name']).union(df['target_name']) for df in simulation_dataframes.values()]))
    simulation_counts = {}

    for sim, df in simulation_dataframes.items():
        df['time'] = pd.to_datetime(df['time'])
        agents = all_agents
        groups = df[['init_name', 'init_group']].drop_duplicates().set_index('init_name')['init_group'].to_dict()
        unique_groups = df['init_group'].unique()
        group_positions = {}
        spacing = 5
        mirror_spacing = 50
        radius = 20

        for i, group in enumerate(unique_groups):
            group_agents = [agent for agent in agents if groups.get(agent) == group]
            np.random.seed(i)
            angle_step = 2 * np.pi / max(len(group_agents), 1)

            for j, agent in enumerate(group_agents):
                angle = j * angle_step
                x = np.cos(angle) * radius
                y = np.sin(angle) * radius
                if group == unique_groups[0]:
                    x += mirror_spacing
                group_positions[agent] = (x, y)

        positions = group_positions

        group_colors = {
            f'{unique_groups[0]}': '#a3c1ff',
            f'{unique_groups[1]}': '#ffcc99'
        }

        edge_colors = {
            'intra_group0': '#6fa3ef',
            'intra_group1': '#f4a460',
            'intergroup': '#bbbbbb'
        }

        df = df.sort_values(by='time')
        G = nx.Graph()
        for agent in agents:
            if agent in groups:
                G.add_node(agent, group=groups[agent])

        frames = []
        last_frame_data = None

        with tempfile.TemporaryDirectory() as temp_dir:
            for index, row in df.iterrows():
                if row['output'] == 'yes':
                    G.add_edge(row['init_name'], row['target_name'])

                fig, ax = plt.subplots(figsize=(14, 10))
                ax.clear()
                colors = [group_colors[G.nodes[node]['group']] for node in G.nodes]
                edge_list = G.edges()
                edge_colors_list = []

                for edge in edge_list:
                    group1 = G.nodes[edge[0]]['group']
                    group2 = G.nodes[edge[1]]['group']
                    if group1 == group2 == unique_groups[0]:
                        edge_colors_list.append(edge_colors['intra_group0'])
                    elif group1 == group2 == unique_groups[1]:
                        edge_colors_list.append(edge_colors['intra_group1'])
                    else:
                        edge_colors_list.append(edge_colors['intergroup'])

                nx.draw(G, pos=positions, ax=ax, with_labels=False, node_size=3500, node_color=colors, edge_color=edge_colors_list, width=2)
                for node, (x, y) in positions.items():
                    ax.text(x, y, node, fontsize=12, ha='center', va='center', color='black')
                ax.set_title(f"Interaction at {row['time']}")
                legend_elements = [
                    Patch(facecolor='#a3c1ff', edgecolor='black', label=f'{unique_groups[0]}'),
                    Patch(facecolor='#ffcc99', edgecolor='black', label=f'{unique_groups[1]}')
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=14)
                plt.tight_layout()

                temp_frame_path = os.path.join(temp_dir, f"frame_{index}.png")
                plt.savefig(temp_frame_path)
                plt.close(fig)
                frames.append(imageio.imread(temp_frame_path))
                last_frame_data = (G.copy(), positions.copy(), colors.copy(), edge_colors_list.copy(), row['time'])

            output_file = f"{simulation_names[sim]}_interaction_timeline.gif"
            
            imageio.mimsave(output_file, frames, duration=0.8)
            print(f"GIF for {simulation_names[sim]} created successfully!")

        if last_frame_data:
            G, positions, colors, edge_colors_list, last_time = last_frame_data
            fig, ax = plt.subplots(figsize=(14, 10))
            nx.draw(G, pos=positions, ax=ax, with_labels=False, node_size=3500, node_color=colors, edge_color=edge_colors_list, width=2)
            for node, (x, y) in positions.items():
                ax.text(x, y, node, fontsize=12, ha='center', va='center', color='black')
            ax.set_title(f"Interaction Map for {simulation_names[sim]}")
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=14)
            plt.tight_layout()
            svg_file = f"{simulation_names[sim]}_final_state.svg"
            plt.savefig(svg_file, format='svg')
            plt.close(fig)
            
            print(f"SVG for {simulation_names[sim]} saved as {svg_file}")
    
    return simulation_counts



def create_birthday_interaction_svg(event_mentions_df, simulation_name):

    # Choose groups based on simulation name.
    if 'village' in simulation_name.lower():
        groups = groups_village
    else:
        groups = groups_race

    # Set up colors for groups and agents.
    unique_groups = list(set(groups.values()))
    group_colors = {
        unique_groups[0]: '#a3c1ff',
        unique_groups[1]: '#ffcc99'
    }
    agent_colors = {agent: group_colors.get(groups.get(agent, 'Unknown'), '#d3d3d3')
                    for agent in groups.keys()}
    
    ordered_nodes = sorted(groups.keys(), key=lambda x: groups[x])
    
    # Compute positions for nodes using a shell layout per group (as in your GIF)
    group_positions = {}
    for i, group in enumerate(unique_groups):
        members = [node for node in ordered_nodes if groups[node] == group]
        shell_pos = nx.shell_layout(nx.complete_graph(len(members)))
        for j, node in enumerate(members):
            # Offset positions so that groups don't overlap
            group_positions[node] = shell_pos[j] + np.array([i * 2, i * 2])
    
    # Build the MultiDiGraph.
    G = nx.MultiDiGraph()
    for agent in ordered_nodes:
        G.add_node(agent)
    
    # Add edges: for each row, add an edge for each birthday mention.
    for index, row in event_mentions_df.iterrows():
        if row['Sam_birthday_mentioned']:
            G.add_edge(row['init_name'], row['target_name'], color='blue')
        if row['Isabella_birthday_mentioned']:
            G.add_edge(row['init_name'], row['target_name'], color='orange')
    
    # Helper function: draw a single curved arrow from src to dst.
    def draw_edge_arrow(ax, pos, src, dst, color, rad, arrowsize):
        arrow = FancyArrowPatch(
            posA=pos[src],
            posB=pos[dst],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="->", 
            mutation_scale=arrowsize, 
            color=color,
            lw=2,
            shrinkA=25,   
            shrinkB=25,
            clip_on=False
        )
        ax.add_patch(arrow)
    
    def draw_multidigraph_edges(G, pos, ax, arrowsize=40):
        seen = set()
        for u, v, key, data in G.edges(keys=True, data=True):
            if (u, v) in seen:
                continue
            edges_data = G.get_edge_data(u, v)
            num_edges = len(edges_data)
            if num_edges == 1:
                key0 = list(edges_data.keys())[0]
                color = edges_data[key0]['color']
                draw_edge_arrow(ax, pos, u, v, color, rad=0.2, arrowsize=arrowsize)
            else:
                offset_range = 0.5
                offsets = np.linspace(-offset_range, offset_range, num_edges)
                sorted_keys = sorted(edges_data.keys())
                for i, k in enumerate(sorted_keys):
                    color = edges_data[k]['color']
                    draw_edge_arrow(ax, pos, u, v, color, rad=offsets[i], arrowsize=arrowsize)
            seen.add((u, v))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    draw_multidigraph_edges(G, pos=group_positions, ax=ax, arrowsize=40)
    
    nx.draw_networkx_nodes(G, pos=group_positions, ax=ax,
                           node_color=[agent_colors[node] for node in G.nodes()],
                           node_size=3000)
    nx.draw_networkx_labels(G, pos=group_positions, ax=ax)
    
    legend_elements = [
        Patch(facecolor=group_colors[group], edgecolor='black', label=f'Group: {group}')
        for group in group_colors
    ] + [
        Patch(facecolor='blue', edgecolor='black', label="Sam's Birthday Mention"),
        Patch(facecolor='orange', edgecolor='black', label="Isabella's Birthday Mention")
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    ax.margins(0.2)
    plt.tight_layout(pad=2)
    
    plt.savefig(f'{simulation_name}_birthday_interaction.svg', format='svg')
    plt.close(fig)


def create_birthday_interaction_gif(event_mentions_dfs, simulation_names):

    for sim, df in event_mentions_dfs.items():
        if 'village' in sim:
            groups = groups_village
        else:
            groups = groups_race

        unique_groups = list(set(groups.values()))
        group_colors = {
            unique_groups[0]: '#a3c1ff',
            unique_groups[1]: '#ffcc99'
        }
        agent_colors = {agent: group_colors.get(groups.get(agent, 'Unknown'), '#d3d3d3') for agent in groups.keys()}

        ordered_nodes = sorted(groups.keys(), key=lambda x: groups[x])
        
        group_positions = {}
        for i, group in enumerate(unique_groups):
            members = [node for node in ordered_nodes if groups[node] == group]
            shell_pos = nx.shell_layout(nx.complete_graph(len(members)))
            for j, node in enumerate(members):
                group_positions[node] = shell_pos[j] + [i * 2, i * 2]
        
        frames = []
        fig, ax = plt.subplots(figsize=(14, 10))  

        G = nx.DiGraph()
        for agent in ordered_nodes:
            G.add_node(agent)

        last_frame_edges = []

        for index, row in df.iterrows():
            if row['Sam_birthday_mentioned']:
                G.add_edge(row['init_name'], row['target_name'], color='blue')
            elif row['Isabella_birthday_mentioned']:
                G.add_edge(row['init_name'], row['target_name'], color='orange')

            ax.clear()
            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            node_colors = [agent_colors[node] for node in G.nodes()]

            nx.draw(
                G,
                pos=group_positions,
                ax=ax,
                with_labels=True,
                node_size=3000,
                edge_color=edge_colors,
                node_color=node_colors,
                connectionstyle="arc3,rad=0.1",
                arrowsize=5
            )

            legend_elements = [
                Patch(facecolor=group_colors[group], edgecolor='black', label=f'Group: {group}') for group in group_colors
            ] + [
                Patch(facecolor='blue', edgecolor='black', label="Sam's Birthday Mention"),
                Patch(facecolor='orange', edgecolor='black', label="Isabella's Birthday Mention")
            ]

            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
            ax.set_title(f"Information spread for {simulation_names[sim]}")

            plt.savefig("temp_frame.png", bbox_inches='tight')
            frames.append(imageio.imread("temp_frame.png"))

            last_frame_edges = list(G.edges(data=True))  

        imageio.mimsave(f'{simulation_names[sim]}_birthday_interaction.gif', frames, duration=1)

        print(f"GIF for {simulation_names[sim]} created successfully!")







def save_agent_conversations(simulation_name, agent_name, simulation_dataframes):
    """
    Extracts all conversations and utterances involving a specified agent from a simulation
    and saves the resulting DataFrame as a CSV file.

    Parameters:
        simulation_name (str): Name of the simulation to process.
        agent_name (str): Name of the agent to filter for.
        simulation_dataframes (dict): Dictionary containing simulation DataFrames.

    Returns:
        None
    """
    # Check if the simulation name exists in the dictionary
    if simulation_name not in simulation_dataframes:
        print(f"Simulation '{simulation_name}' not found.")
        return

    # Get the DataFrame for the specified simulation
    df = simulation_dataframes[simulation_name]

    # Filter rows where the agent is either the init_name or target_name
    filtered_df = df[(df['init_name'] == agent_name) | (df['target_name'] == agent_name)]

    # Define the file path to save the filtered DataFrame
    file_path = os.path.join(os.getcwd(), f"{simulation_name}_{agent_name}_conversations.csv")

    # Save the filtered DataFrame to a CSV file
    filtered_df.to_csv(file_path, index=False)

    print(f"DataFrame saved to: {file_path}")




def compute_termination_matrix(conv_summary, ordered_agents):
    """
    Compute a matrix where each entry (agent1 -> agent2) represents the 
    percentage of conversations between the two agents that were ended by agent1.

    Returns:
    - termination_matrix: A matrix where each cell represents the percentage 
      of conversations ended by the row agent when conversing with the column agent.
    """

    # Initialize the termination matrix
    termination_matrix = pd.DataFrame(np.nan, index=ordered_agents, columns=ordered_agents)

    # Dictionary to track conversation statistics for each agent pair
    conv_counts = {(a1, a2): {'total': 0, 'end_by_a1': 0, 'end_by_a2': 0} 
                   for a1 in ordered_agents for a2 in ordered_agents}

    # Process each conversation
    for _, row in conv_summary.iterrows():
        agent1 = row['init_name']
        agent2 = row['target_name']


        # Ensure agent1 and agent2 are distinct
        if agent1 == agent2:
            continue  # Skip self-conversations

        # Store conversation pair
        pair = (agent1, agent2)
        reverse = (agent2, agent1)

        # Count total interactions between these two agents
        conv_counts[pair]['total'] += 1
        conv_counts[reverse]['total'] += 1

        # Only count conversations that ended early (before reaching full length)
        if row['utterance_length'] < 15:
            conv_counts[pair]['end_by_a1'] += 1  # Always count as ended by init_name
            conv_counts[reverse]['end_by_a2'] +=1
        else:
            continue  # Conversations that reach full length are ignored for termination count

    # Compute percentages for each agent pair
    for (agent1, agent2), stats in conv_counts.items():
        total_conv = stats['total'] 
        if total_conv > 0:
            termination_matrix.at[agent1, agent2] = stats['end_by_a1'] / total_conv
            termination_matrix.at[agent2, agent1] = stats['end_by_a2'] / total_conv

    return termination_matrix

def compute_matrices(simulation_dataframes, ordered_agents, groups_race, know_each_other = True):
    """Compute and return relative score matrices for each simulation."""

    relative_score_matrices = {}

    for sim_name, df in simulation_dataframes.items():
        df_filtered = df[df['know_each_other'] == know_each_other]

        # Count interactions by init_name and target_name
        interaction_counts = df_filtered.groupby(['init_name', 'target_name', 'output']).size().reset_index(name='count')

        # Create all possible combinations of agents and outputs
        all_combinations = pd.MultiIndex.from_product(
            [ordered_agents, ordered_agents, ['yes', 'no']],
            names=['init_name', 'target_name', 'output']
        )

        # Reindex the dataframe to include all combinations and fill missing counts with 0
        interaction_counts = interaction_counts.set_index(['init_name', 'target_name', 'output']).reindex(all_combinations, fill_value=0).reset_index()

        # Pivot to create ordered matrices
        matrix_yes = interaction_counts[interaction_counts['output'] == 'yes'].pivot(index='init_name', columns='target_name', values='count').fillna(0)
        matrix_no = interaction_counts[interaction_counts['output'] == 'no'].pivot(index='init_name', columns='target_name', values='count').fillna(0)

        # Ensure row & column order is preserved
        matrix_yes = matrix_yes.reindex(index=ordered_agents, columns=ordered_agents, fill_value=0)
        matrix_no = matrix_no.reindex(index=ordered_agents, columns=ordered_agents, fill_value=0)

        # Compute the new relative score
        matrix_relative_score = (matrix_yes - matrix_no) / (matrix_yes + matrix_no)
        matrix_relative_score[matrix_yes + matrix_no == 0] = np.nan  # Hide cases with no interaction

        # Store the relative score matrix for each simulation
        relative_score_matrices[sim_name] = matrix_relative_score

    return relative_score_matrices


def shorten_village_labels(label):
    """Replace long village labels with shorter versions."""
    label = label.replace("newcomer to the village", "newcomer")
    label = label.replace("longterm resident of the village", "long-term res.")
    return label

def process_and_plot_simulation_data_perc(relative_score_matrices, ordered_agents, village_simulations, race_simulations, groups_village, groups_race, shorten_village_labels):
    """
    Processes simulation matrices and generates boxplots for village and race simulations.
    
    Parameters:
    - relative_score_matrices: dict of simulation matrices
    - ordered_agents: list of agents in order
    - village_simulations: set of village simulation names
    - race_simulations: set of race simulation names
    - groups_village: mapping of agents to village groups
    - groups_race: mapping of agents to race groups
    - shorten_village_labels: function to shorten village labels
    """
    # Prepare data storage
    boxplot_data_village = []
    boxplot_data_race = []

    # Process each simulation matrix
    for sim_name, matrix in relative_score_matrices.items():
        for i, agent1 in enumerate(ordered_agents):
            for j, agent2 in enumerate(ordered_agents):
                if i == j:  # Skip self-interactions
                    continue
                value = matrix.iloc[i, j]
                if np.isnan(value):
                    continue

                # Determine which simulation category this belongs to
                if sim_name in village_simulations:
                    category = "Village Simulation"
                    group_mapping = groups_village
                elif sim_name in race_simulations:
                    category = "Race Simulation"
                    group_mapping = groups_race
                else:
                    continue  # Skip if the simulation is unrecognized
                # Identify interaction type and shorten labels for village simulations
                init_group = group_mapping.get(agent1, "Unknown")
                target_group = group_mapping.get(agent2, "Unknown")

                if category == "Village Simulation":
                    init_group = shorten_village_labels(init_group)
                    target_group = shorten_village_labels(target_group)

                interaction_type = f"{init_group} → {target_group}"

                # Store data
                new_entry = {
                    "Simulation": sim_name,
                    "Category": category,
                    "Interaction Type": interaction_type,
                    "Relative Score": value
                }

                if category == "Village Simulation":
                    boxplot_data_village.append(new_entry)
                else:
                    boxplot_data_race.append(new_entry)

    # Convert to DataFrames
    boxplot_df_village = pd.DataFrame(boxplot_data_village)
    boxplot_df_race = pd.DataFrame(boxplot_data_race)

    # Define palettes:
    box_palette = sns.color_palette("Set2", 2)
    point_palette = sns.color_palette("Set1", 2)

    # --- Village Simulation Plot ---
    plt.figure(figsize=(12, 6))
    
    # Plot boxplot using the Set2 palette
    ax_village = sns.boxplot(
        data=boxplot_df_village, 
        x="Interaction Type", 
        y="Relative Score", 
        hue="Simulation", 
        palette=box_palette
    )
    if ax_village.get_legend() is not None:
        ax_village.get_legend().remove()

    # Plot pointplot using the Set1 palette
    sns.pointplot(
        data=boxplot_df_village, 
        x="Interaction Type", 
        y="Relative Score", 
        hue="Simulation",
        dodge=0.5, 
        linestyle='none', 
        markers="o", 
        capsize=0.2, 
        err_kws={'linewidth': 1}, 
        palette=point_palette,
        errorbar='se'
    )
    
    # Formatting the plot
    plt.axhline(y=0, linestyle="dashed", color="black", alpha=0.6)
    plt.title("Village Simulations: Percentage of conversations terminated before maximum length")
    plt.xlabel("Interaction Type")
    plt.ylabel("Percentage")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    
    # Remove any automatic legend (if present)
    leg = plt.gca().get_legend()
    if leg is not None:
        leg.remove()

    # Create composite legend handles for each simulation.
    sim1_box = mpatches.Patch(color=box_palette[0])
    sim1_point = mlines.Line2D([], [], color=point_palette[0], marker='o', linestyle='None', markersize=8)
    sim2_box = mpatches.Patch(color=box_palette[1])
    sim2_point = mlines.Line2D([], [], color=point_palette[1], marker='o', linestyle='None', markersize=8)

    # Create the composite handles as tuples
    handles = [(sim1_box, sim1_point), (sim2_box, sim2_point)]
    labels = ["Simulation 1", "Simulation 2"]

    plt.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=2)}, title="Simulation")
    
    # Save and show the village plot
    plt.savefig("village_perc_plot.svg", format='svg')
    plt.show()

    # --- Race Simulation Plot ---
    plt.figure(figsize=(12, 6))
    
    ax_race = sns.boxplot(
        data=boxplot_df_race, 
        x="Interaction Type", 
        y="Relative Score", 
        hue="Simulation", 
        palette=box_palette
    )
    if ax_race.get_legend() is not None:
        ax_race.get_legend().remove()

    sns.pointplot(
        data=boxplot_df_race, 
        x="Interaction Type", 
        y="Relative Score", 
        hue="Simulation",
        dodge=0.5, 
        linestyle='none', 
        markers="o", 
        capsize=0.2, 
        err_kws={'linewidth': 1}, 
        palette=point_palette,
        errorbar='se'
    )
    
    plt.axhline(y=0, linestyle="dashed", color="black", alpha=0.6)
    plt.title("Race Simulations: Percentage of conversations terminated before maximum length")
    plt.xlabel("Interaction Type")
    plt.ylabel("Relative Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    
    leg = plt.gca().get_legend()
    if leg is not None:
        leg.remove()


    plt.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=2)}, title="Simulation")
    
    plt.savefig("race_perc_plot.svg", format='svg')
    plt.show()



    return boxplot_df_village, boxplot_df_race



def process_and_plot_simulation_data(relative_score_matrices, ordered_agents, village_simulations, race_simulations, groups_village, groups_race, shorten_village_labels, known=False):
    """
    Processes simulation matrices and generates boxplots for village and race simulations.
    
    Parameters:
    - relative_score_matrices: dict of simulation matrices
    - ordered_agents: list of agents in order
    - village_simulations: set of village simulation names
    - race_simulations: set of race simulation names
    - groups_village: mapping of agents to village groups
    - groups_race: mapping of agents to race groups
    - shorten_village_labels: function to shorten village labels
    - known: bool flag to switch titles and filenames
    """
    # Prepare data storage
    boxplot_data_village = []
    boxplot_data_race = []

    # Process each simulation matrix
    for sim_name, matrix in relative_score_matrices.items():
        for i, agent1 in enumerate(ordered_agents):
            for j, agent2 in enumerate(ordered_agents):
                if i == j:  # Skip self-interactions
                    continue
                value = matrix.iloc[i, j]
                if np.isnan(value):
                    continue

                # Determine which simulation category this belongs to
                if sim_name in village_simulations:
                    category = "Village Simulation"
                    group_mapping = groups_village
                elif sim_name in race_simulations:
                    category = "Race Simulation"
                    group_mapping = groups_race
                else:
                    continue 

                # Identify interaction type and shorten labels for village simulations
                init_group = group_mapping.get(agent1, "Unknown")
                target_group = group_mapping.get(agent2, "Unknown")

                if category == "Village Simulation":
                    init_group = shorten_village_labels(init_group)
                    target_group = shorten_village_labels(target_group)

                interaction_type = f"{init_group} → {target_group}"

                # Store data
                new_entry = {
                    "Simulation": sim_name,
                    "Category": category,
                    "Interaction Type": interaction_type,
                    "Relative Score": value
                }

                if category == "Village Simulation":
                    boxplot_data_village.append(new_entry)
                else:
                    boxplot_data_race.append(new_entry)

    # Convert to DataFrames
    boxplot_df_village = pd.DataFrame(boxplot_data_village)
    boxplot_df_race = pd.DataFrame(boxplot_data_race)

    # Define palettes: Set2 for the boxplots and Set1 for the pointplots
    box_palette = sns.color_palette("Set2", 2)
    point_palette = sns.color_palette("Set1", 2)

    # --- Village Simulation Plot ---
    plt.figure(figsize=(12, 6))
    # Boxplot with Set2 palette
    ax = sns.boxplot(
        data=boxplot_df_village, 
        x="Interaction Type", 
        y="Relative Score", 
        hue="Simulation", 
        palette=box_palette
    )
    # Pointplot with Set1 palette
    sns.pointplot(
        data=boxplot_df_village, 
        x="Interaction Type", 
        y="Relative Score", 
        hue="Simulation",
        dodge=0.5, 
        linestyle='none', 
        markers="o", 
        capsize=0.2, 
        err_kws={'linewidth': 1}, 
        palette=point_palette,
        errorbar='se'
    )
    plt.axhline(y=0, linestyle="dashed", color="black", alpha=0.6)
    
    if known:
        plt.title("Decision to Initiate Conversations: Yes/No Ratio in Village Simulations (Acquainted)")
    else: 
        plt.title("Decision to Initiate Conversations: Yes/No Ratio in Village Simulations (Unacquainted)")
    
    plt.xlabel("Interaction Type")
    plt.ylabel("Yes/No Ratio")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    leg = plt.gca().get_legend()
    if leg is not None:
        leg.remove()

    # Create composite legend handles:
    sim1_box = mpatches.Patch(color=box_palette[0])
    sim1_point = mlines.Line2D([], [], color=point_palette[0], marker='o', linestyle='None', markersize=8)
    sim2_box = mpatches.Patch(color=box_palette[1])
    sim2_point = mlines.Line2D([], [], color=point_palette[1], marker='o', linestyle='None', markersize=8)
    composite_handles = [(sim1_box, sim1_point), (sim2_box, sim2_point)]
    composite_labels = ["1st Simulation", "2nd Simulation"]

    plt.legend(composite_handles, composite_labels, handler_map={tuple: HandlerTuple(ndivide=2)}, title="Simulation", loc='lower left')
    
    if known: 
        plt.savefig("village_decSpeak_plot.svg", format='svg')
    else:
        plt.savefig("village_decSpeak_notKnow_plot.svg", format='svg')
    plt.show()

    # --- Race Simulation Plot ---
    plt.figure(figsize=(12, 6))
    # Boxplot with Set2 palette
    ax = sns.boxplot(
        data=boxplot_df_race, 
        x="Interaction Type", 
        y="Relative Score", 
        hue="Simulation", 
        palette=box_palette
    )
    # Pointplot with Set1 palette
    sns.pointplot(
        data=boxplot_df_race, 
        x="Interaction Type", 
        y="Relative Score", 
        hue="Simulation",
        dodge=0.5, 
        linestyle='none', 
        markers="o", 
        capsize=0.2, 
        err_kws={'linewidth': 1}, 
        palette=point_palette,
        errorbar='se'
    )
    plt.axhline(y=0, linestyle="dashed", color="black", alpha=0.6)
    if known:
        plt.title("Decision to Initiate Conversations: Yes/No Ratio in Race Simulations (Acquainted)")
    else:
        plt.title("Decision to Initiate Conversations: Yes/No Ratio in Race Simulations (Unacquainted)")
    plt.xlabel("Interaction Type")
    plt.ylabel("Yes/No Ratio")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    leg = plt.gca().get_legend()
    if leg is not None:
        leg.remove()

    plt.legend(composite_handles, composite_labels, handler_map={tuple: HandlerTuple(ndivide=2)}, title="Simulation", loc='upper right')
    
    if known:
        plt.savefig("race_decSpeak_plot.svg", format='svg')
    else:
        plt.savefig("race_decSpeak_notKnow_plot.svg", format='svg')
    plt.show()

    return boxplot_df_village, boxplot_df_race



def compute_summary_statistics(df):
    """Compute summary statistics including mean, median, standard error, IQR, quartiles"""
    
    def iqr(x):
        return np.percentile(x, 75) - np.percentile(x, 25)  

    def q1(x):
        return np.percentile(x, 25) 
    
    def q3(x):
        return np.percentile(x, 75)  
    
    
    summary_stats = df.groupby(["Interaction Type", "Simulation"])["Relative Score"].agg(
        Median="median",
        Mean="mean",
        Std_Error=lambda x: np.std(x, ddof=1) / np.sqrt(len(x)), 
        IQR=iqr, 
        Q1=q1,  
        Q3=q3,  
    ).reset_index()
    
    return summary_stats


def plot_heatmaps(simulation_dataframes, ordered_agents, groups_race, know_each_other = True):

    """Generate heatmaps for interaction counts and relative scores."""

    for sim_name, df in simulation_dataframes.items():
        df_filtered = df[df['know_each_other'] == know_each_other]

       
        interaction_counts = df_filtered.groupby(['init_name', 'target_name', 'output']).size().reset_index(name='count')

        # Create all possible combinations of agents and outputs
        all_combinations = pd.MultiIndex.from_product(
            [ordered_agents, ordered_agents, ['yes', 'no']],
            names=['init_name', 'target_name', 'output']
        )

        # Reindex the dataframe to include all combinations and fill missing counts with 0
        interaction_counts = interaction_counts.set_index(['init_name', 'target_name', 'output']).reindex(all_combinations, fill_value=0).reset_index()

        # Pivot to create ordered matrices
        matrix_yes = interaction_counts[interaction_counts['output'] == 'yes'].pivot(index='init_name', columns='target_name', values='count').fillna(0)
        matrix_no = interaction_counts[interaction_counts['output'] == 'no'].pivot(index='init_name', columns='target_name', values='count').fillna(0)

        matrix_yes = matrix_yes.reindex(index=ordered_agents, columns=ordered_agents, fill_value=0)
        matrix_no = matrix_no.reindex(index=ordered_agents, columns=ordered_agents, fill_value=0)

        # Compute the ratio score
        matrix_relative_score = (matrix_yes - matrix_no) / (matrix_yes + matrix_no)
        matrix_relative_score[matrix_yes + matrix_no == 0] = np.nan  

        split_index = next(i for i, agent in enumerate(ordered_agents) if groups_race[agent] == "White")

        matrices = {
            "Absolute Counts (Yes)": (matrix_yes, "Blues"),
            "Absolute Counts (No)": (matrix_no, "Oranges"),
            "Relative Score": (matrix_relative_score, "RdBu_r")
        }

        # Plot heatmaps with boundary lines
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        heatmaps = {
            "Absolute Counts (Yes)": (matrix_yes, "Blues", axes[0]),
            "Absolute Counts (No)": (matrix_no, "Oranges", axes[1]),
            "Relative Score": (matrix_relative_score, "RdBu_r", axes[2])
        }

        for title, (matrix, cmap, ax) in heatmaps.items():
            sns.heatmap(matrix, annot=True, fmt=".2f" if title == "Relative Score" else "d", cmap=cmap, center=0 if title == "Relative Score" else None, ax=ax)
            ax.set_title(f"{sim_name} - {title}")
            ax.axvline(x=split_index, color='black', linewidth=2)  # Vertical line
            ax.axhline(y=split_index, color='black', linewidth=2)  # Horizontal line
        plt.show()


