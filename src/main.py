import pandas as pd
from src.cell_graph import CellGraph
from src.mcmc_tracker import MCMCTracker


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("data/multisectionmeasurements.csv")
    sections = [
        CellGraph(
            coords=section_data[['Centroid X µm', 'Centroid Y µm']].values,
            areas=section_data['Area'].values,
            perimeters=section_data['Perimeter'].values,
            distances=section_data['Distance in um to nearest Cell'].values,
            names=section_data['Name'].values
        )
        for _, section_data in data.groupby('Image')
    ]

    # Run MCMC Tracker
    tracker = MCMCTracker(sections)
    tracker.run(max_iter=3)

    # Save results
    tracker.save_results("output/tracking_results.csv")