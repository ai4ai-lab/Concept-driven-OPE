import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_gridworld_env_description(env, filename='environment_description_detailed.pdf'):
    fig, ax = plt.subplots(figsize=(8,8))

    # Define the annotations for wind and penalty based on sector data

    wind_info = {
            (0, 4, 0, 4): ('', 0),
            (4, 8, 0, 4): ('→↓ (+1,-1)', -2),
            (8, 12, 0, 4): ('↓ (+0,-1)', -1),
            (12, 16, 0, 4): ('↓ (+0,-1)', -1),
            (16, 20, 0, 4): ('↓ (+0,-1)', -1),
            (0, 4, 4, 8): ('', 0),
            (4, 8, 4, 8): ('', 0),
            (8, 12, 4, 8): ('→↓ (+1,-1)', -2),
            (12, 16, 4, 8): ('↓ (+0,-1)', -1),
            (16, 20, 4, 8): ('↓ (+0,-1)', -1),
            (0, 4, 8, 12): ('←↑ (-1,+1)', -2),
            (4, 8, 8, 12): ('', 0),
            (8, 12, 8, 12): ('', 0),
            (12, 16, 8, 12): ('→↓ (+1,-1)', -2),
            (16, 20, 8, 12): ('↓ (+0,-1)', -1),
            (0, 4, 12, 16): ('← (-1)', -1),
            (4, 8, 12, 16): ('←↑ (-1,+1)', -2),
            (8, 12, 12, 16): ('', 0),
            (12, 16, 12, 16): ('', 0),
            (16, 20, 12, 16): ('→↓ (+1,-1)', -2),
            (0, 4, 16, 20): ('← (-1)', -1),
            (4, 8, 16, 20): ('← (-1)', -1),
            (8, 12, 16, 20): ('←↑ (-1,+1)', -2),
            (12, 16, 16, 20): ('', 0),
            (16, 20, 16, 20): ('', 0)
    }

    # Plot the grid regions and annotate with wind and penalties
    for (x_min, x_max, y_min, y_max), (wind, penalty) in wind_info.items():
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='black', facecolor='white')
        ax.add_patch(rect)

        # Font size for wind direction proportional to the penalty magnitude
        font_size = 10
        penalty_text = f'Penalty: {penalty}' if penalty != 0 else 'Penalty: 0'
        wind_text = f'Wind: {wind}' if wind else ''

        # Combine texts and display
        ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, f'{wind_text}\n{penalty_text}', color='black', ha='center', va='center', fontsize=font_size)

    # Add the target region in the top right corner
    target_rect = patches.Rectangle((19, 19), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(target_rect)

    # Set grid settings
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(range(0, 21, 4))
    ax.set_yticks(range(0, 21, 4))
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Save the plot as a PDF file
    plt.savefig(filename, dpi=200)
    plt.show()

# Call the function with the environment description
plot_gridworld_env_description(env, filename='environment_description.pdf')
