import requests
import json
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from geopy.distance import distance
import numpy as np
import folium
import math
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
import os
from PyPDF2 import PdfReader, PdfWriter
# from Route_Classes import *

class Assignment:
    def __init__(self, MAX_CLUSTER_SIZE, MAX_WALKING_TIME, MAX_CLUSTER_DISTANCE) -> None:
        self.__NUM_CLUSTERS = 3
        self.__MAX_CLUSTER_SIZE = MAX_CLUSTER_SIZE
        self.__MAX_CLUSTER_DISTANCE = MAX_CLUSTER_DISTANCE # in km
        self.__MAX_WALKING_TIME = MAX_WALKING_TIME # in minutes
        self.__dataset = None
        self.__is_clustered = False
        self.__is_geocoded = False
        self.__clusters = None
        self.__num_locations = 0
        self.__max_intra_cluster_distance = None
        self.__widest_cluster = None
        self.__cluster_map = None
        self.__optimised_routes = None

    def get_optimised_routes(self):
        return self.__optimised_routes

    def get_num_clusters(self):
        return self.__NUM_CLUSTERS
    
    def get_num_locations(self):
        return self.__num_locations

    def get_max_cluster_size(self):
        return self.__MAX_CLUSTER_SIZE
    
    def set_max_cluster_size(self, max_cluster_size):
        if isinstance(max_cluster_size, int):
            self.__MAX_CLUSTER_SIZE = max_cluster_size
            return
        print('Error: Max. cluster size should be an integer')
        
    def get_max_cluster_distance(self):
        return self.__MAX_CLUSTER_DISTANCE
    
    def set_max_cluster_distance(self, MAX_CLUSTER_DISTANCE):
        if isinstance(MAX_CLUSTER_DISTANCE, float) or isinstance(MAX_CLUSTER_DISTANCE, int):
            self.__MAX_CLUSTER_DISTANCE = MAX_CLUSTER_DISTANCE
            return
        print('Error: Max cluster distance should be a float/integer in km')
    
    def get_max_walking_time(self):
        return self.__MAX_WALKING_TIME

    def set_max_walking_time(self, MAX_WALKING_TIME):
        if isinstance(MAX_WALKING_TIME, float) or isinstance(MAX_WALKING_TIME, int):
            self.__MAX_WALKING_TIME = MAX_WALKING_TIME
            return
        print('Error: Max walking time should be a float/integer in minutes')
    
    def upload_csv(self, file_path):
        dataset = pd.read_csv(file_path)
        dataset.columns = dataset.columns.str.title()
        self.__dataset = dataset
        self.__num_locations = len(dataset)
        self.__is_clustered = False
        self.__is_geocoded = False
    
    def upload_excel(self, file_path):
        dataset = pd.read_excel(file_path)
        dataset.columns = dataset.columns.str.title()
        self.__dataset = dataset
        self.__num_locations = len(dataset)
        self.__is_clustered = False
        self.__is_geocoded = False
    
    def get_dataset(self):
        print(f'Dataset configuration\nGeocoded: {self.__is_geocoded}\nClustered: {self.__is_clustered}')
        return self.__dataset

    def get_clusters(self):
        return self.__clusters

    def geocode_dataset(self):
        if self.__is_geocoded:
            print('Dataset is already geocoded')
            return
        latitudes = []
        longitudes = []
        full_addresses = []
        X = []
        Y = []
        get_address_details = 'Y'
        return_geom = 'Y'
        for i, row in self.__dataset.iterrows():
            # Form address
            if math.isnan(row['Storey']):
                address = f'{row["Block Number"]} {row["Street Name"]} SINGAPORE {row["Postal Code"]:06}'
            else:
                address = f'{row["Block Number"]} {row["Street Name"]} {row["Storey"]:02}-{row["Unit Number"]:02} SINGAPORE {row["Postal Code"]:06}'
            
            # Create URL to get Geolocation
            postal_code = f'{row["Postal Code"]:06}'
            url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal='+{postal_code}'&returnGeom={return_geom}&getAddrDetails={get_address_details}&pageNum=1"
            response = requests.request('GET', url).text
            response = json.loads(response)
            # print(response)
            if response['found'] == 0:
                print(f'Address {i + 1}: {address} cannot be found')
                return
            response = response['results']
            latitude = float(response[0]['LATITUDE'])
            longitude = float(response[0]['LONGITUDE'])
            x = float(response[0]['X'])
            y = float(response[0]['Y'])
            X.append(x)
            Y.append(y)
            latitudes.append(latitude)
            longitudes.append(longitude)
            full_addresses.append(address)
        new_fields = pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes, 'X': X, 'Y': Y, 'Full Address': full_addresses})
        self.__dataset = pd.concat([self.__dataset, new_fields], axis=1)
        self.__is_geocoded = True
        return

    def cluster_dataset(self):
        counts = self._cluster_helper(self.__dataset)
        max_distance, widest_cluster = self._check_cluster_size(self.__dataset)
        while(max_distance > self.__MAX_CLUSTER_DISTANCE or max(counts) > self.__MAX_CLUSTER_SIZE):
            self.__NUM_CLUSTERS += 1
            counts = self._cluster_helper(self.__dataset)
            max_distance, widest_cluster = self._check_cluster_size(self.__dataset)
        print('Clustering completed')
        print(f'Total number of locations: {len(self.__dataset)}')
        print(f'Total number of clusters: {self.__NUM_CLUSTERS}')
        print(f'Cluster {widest_cluster} has the largest intra-cluster distance of {max_distance} km')
        self.__widest_cluster = widest_cluster
        self.__max_intra_cluster_distance = max_distance
        self.__is_clustered = True
        self._create_all_clusters(self.__dataset)
        
    def _check_cluster_size(self, df):
        max_distance = 0
        widest_cluster = None
        for i in range(self.__NUM_CLUSTERS):
            current_cluster = df[df['cluster'] == i]
            matrix = []
            for index1, row1 in current_cluster.iterrows():
                distance_row = []
                for index2, row2 in current_cluster.iterrows():
                    coords_1 = (row1['Latitude'], row1['Longitude'])
                    coords_2 = (row2['Latitude'], row2['Longitude'])
                    distance_row.append(distance(coords_1, coords_2).kilometers)
                matrix.append(distance_row)
            matrix_max_distance = max(max(matrix))
            if matrix_max_distance > max_distance:
                max_distance = matrix_max_distance
                widest_cluster = i
        return max_distance, widest_cluster

    def _cluster_helper(self, df):
        X = df[['X', 'Y']]
        ac = AgglomerativeClustering(self.__NUM_CLUSTERS)
        df['cluster'] = ac.fit_predict(X)
        counts = np.bincount(df['cluster'])
        return counts

    def _create_all_clusters(self, df):
        clusters = []
        for i in range(self.__NUM_CLUSTERS):
            current_cluster = df[df['cluster'] == i]
            cluster = self._create_cluster(current_cluster, i)
            clusters.append(cluster)
        self.__clusters = clusters

    def _create_cluster(self, df, cluster_num):
        locations = []
        for i, row in df.iterrows():
            # address = Address(row['Block Number'], row['Street Name'], row['Storey'], row['Unit Number'], row['Postal Code'], row['Full Address'], row['Latitude'], row['Longitude'], row['X'], row['Y'])
            locations.append(row['Full Address'])
        return Cluster(locations, cluster_num, self.__MAX_WALKING_TIME)

    def plot_clusters(self):
        plt.figure(figsize=(10, 6))
        num_clusters = len(self.__clusters)
        colormap = plt.get_cmap('tab20', num_clusters)
        for cluster_num in range(num_clusters):
            color = colormap(cluster_num)
            label = f'Cluster {cluster_num + 1}'
            clustered_data = self.__dataset[self.__dataset['cluster'] == cluster_num]
            plt.scatter(clustered_data['Longitude'], clustered_data['Latitude'], c=[color], label=label, alpha=0.6)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Agglomerative Clustering of Addresses in Singapore\
                  \nNo. of clusters = {len(self.__clusters)}\
                  \nWidest cluster = {self.__widest_cluster + 1}\
                  \nMax intra-cluster distance = {self.__max_intra_cluster_distance:.2f} km')
        plt.legend(bbox_to_anchor=[1, 1], loc='upper left')
        plt.savefig('./cluster_plot/cluster plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.__dataset.to_csv('New_test.csv')

    def plot_map_cluster(self, cluster_num):
        # Remember to change from 1-index to 0-index
        dataset = self.__optimised_routes[cluster_num]
        print(f"\nplotting dataset: {dataset}")
        colormap = plt.get_cmap('tab20', len(self.__clusters))
        color = colormap(cluster_num)
        color = [int(c * 255) for c in color[:3]]
        color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        lat = dataset.iloc[0]['Start Latitude']
        lng = dataset.iloc[0]['Start Longitude']
        cluster_map = folium.Map(location = [lat, lng], zoom_start=11)
        for i, row in dataset.iterrows():
            start_lat = row['Start Latitude']
            start_long = row['Start Longitude']
            start_lat_long = (start_lat, start_long)
            end_lat = row['End Latitude']
            end_long = row['End Longitude']
            end_lat_long = (end_lat, end_long)
            marker = folium.CircleMarker(
                location = start_lat_long,
                radius=10,
                weight=1,
                fill=True,
                fill_color=color_hex,
                color=color_hex,
                fill_opacity=0.8,
                tooltip=folium.Tooltip(f"{i + 1}", permanent=True),
                popup=f"{row['Start Address']}"
            ).add_to(cluster_map)
            folium.PolyLine(locations=[start_lat_long, end_lat_long], color='blue', dash_array='5, 5').add_to(cluster_map)
        if dataset.iloc[0]['Start Latitude'] != dataset.iloc[-1]['End Latitude'] or dataset.iloc[0]['Start Longitude'] != dataset.iloc[-1]['End Longitude']:
            end_lat = dataset.iloc[-1]['End Latitude']
            end_long = dataset.iloc[-1]['End Longitude']
            end_lat_long = (end_lat, end_long)
            marker = folium.CircleMarker(
                location = end_lat_long,
                radius=10,
                weight=1,
                fill=True,
                fill_color=color_hex,
                color=color_hex,
                fill_opacity=0.8,
                tooltip=folium.Tooltip(f"{len(dataset) + 1}", permanent=True),
                popup=f"{dataset.iloc[-1]['End Address']}"
            ).add_to(cluster_map)
        cluster_map.save(f'./cluster_maps/cluster_{cluster_num + 1}.html')
        
    def plot_map_all_cluster(self):
        # Remember to change from 1-index to 0-index
        colormap = plt.get_cmap('tab20', len(self.__clusters))
        lat = self.__dataset.iloc[0]['Latitude']
        lng = self.__dataset.iloc[0]['Longitude']
        cluster_map = folium.Map(location = [lat, lng], zoom_start=11)
        for _, row in self.__dataset.iterrows():
            cluster_id = row['cluster']
            color = colormap(cluster_id)
            color = [int(c * 255) for c in color[:3]]
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            folium.CircleMarker(
                location = (row["Latitude"], row["Longitude"]),
                radius=10,
                weight=4,
                fill=False,
                fill_color=color_hex,
                color='black',
                fill_opacity=0.5,
                tooltip=f"Cluster {cluster_id + 1}\n{row['Full Address']}"
            ).add_to(cluster_map)
        cluster_map.save(f'./cluster_maps/cluster_combined.html')
        self.__cluster_map = cluster_map
        return cluster_map

    def get_cluster_map(self):
        return self.__cluster_map

    def assign_surveyor(self, cluster_num, surveyor_name, start_location, end_location):
        if cluster_num >= len(self.__clusters):
            print('Error: The number exceeds the number of clusters')
            return
        elif cluster_num < 0:
            print('Error: Invalid cluster number')
            return
        cluster = self.__clusters[cluster_num]
        cluster.set_surveyor(surveyor_name, start_location, end_location)
        print(cluster)

    def assign_all_surveyors(self, surveyor_names, start_locations, end_locations):
        if not (len(surveyor_names) == len(start_locations) == len(end_locations) == len(self.__clusters)):
            print('Error: The number of surveyors and their home addresses do not match the number of clusters')
            return
        for i, (surveyor_name, start_location, end_location) in enumerate(zip(surveyor_names, start_locations, end_locations)):
            self.assign_surveyor(i, surveyor_name, start_location, end_location)
        return

    def optimise_all_routes(self, api_key):
        optimised_routes = []
        for cluster in self.__clusters:
            optimised_route_df = cluster.optimise_route(api_key)
            optimised_routes.append(optimised_route_df)
        self.__optimised_routes = optimised_routes
        print('Routes optimised')

    def generate_all_pdfs_and_maps(self):
        clusters = self.get_clusters()
        for i in range(len(self.get_optimised_routes())):
            cluster = clusters[i]
            self.plot_map_cluster(i)
            cluster.generate_pdf()

    def generate_combined_pdf(self):
        pdf_writer = PdfWriter()
        folder_path = './pdfs'
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.pdf'):
                file_path = os.path.join(folder_path, filename)
                pdf_reader = PdfReader(file_path)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    pdf_writer.add_page(page)
        with open("./pdfs/combined_survey_locations.pdf", 'wb') as output_pdf:
            pdf_writer.write(output_pdf)

    def generate_combined_df(self):
        combined_df = []
        for cluster in self.__clusters:
            combined_df.append(cluster.save_route_df())
        combined_df = pd.concat(combined_df, ignore_index=True)
        combined_df.to_excel('./clustercsv/combined_cluster_data.xlsx')
        return

class Cluster:
    def __init__(self, locations, cluster_num, MAX_WALKING_TIME):
        if not (isinstance(locations, list)):
            print('Error: locations must be in a list')
        elif not (isinstance(cluster_num, int)):
            print('Error: The cluster number must be an integer')
        self.__cluster_num = cluster_num
        self.__num_locations = len(locations)
        self.__locations = locations
        self.__MAX_WALKING_TIME = MAX_WALKING_TIME # Minutes
        self.__is_optimised = False
        self.__surveyor = None
        self.__start_address = None
        self.__end_address = None
        self.__route = None
        self.__route_df = None

    def get_cluster_num(self):
        return self.__cluster_num
    
    def set_cluster_num(self, cluster_num):
        self.__cluster_num = cluster_num

    def get_num_locations(self):
        return self.__num_locations

    def get_surveyor(self):
        return self.__surveyor
    
    def get_start_address(self):
        return self.__start_address
    
    def get_end_address(self):
        return self.__end_address
    
    def set_surveyor(self, name, start_address='18 Havelock Road', end_address='18 Havelock Road'):
        if start_address == '':
            start_address='18 Havelock Road'
        if end_address == '':
            end_address='18 Havelock Road'
        self.__surveyor = name
        self.__start_address = start_address
        self.__end_address = end_address
        self.__is_optimised = False
    
    def get_locations(self):
        return self.__locations
    
    def set_locations(self, location):
        self.__locations = location
        self.__is_optimised = False
        self.__route = None
        print('Remember to re-optimise route')
    
    def optimise_route(self, api_key):
        if not self.__start_address or not self.__end_address:
            print('Error: Set the surveyor and start address using the .set_surveyor method')
            return
        if self.__is_optimised:
            print('Route is already optimised')
            return
        origin = self.__start_address
        destination = self.__end_address
        waypoints = self.__locations
        waypoints_str = '|'.join(waypoints)
        mode='walking'
        url = f'https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&mode={mode}&waypoints=optimize:true|{waypoints_str}&key={api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            directions = response.json()
            if directions['status'] == 'OK':
                routes = directions['routes'][0]
                start_lat = [leg["start_location"]["lat"] for leg in routes["legs"]]
                start_long = [leg["start_location"]["lng"] for leg in routes["legs"]]
                end_lat = [leg["end_location"]["lat"] for leg in routes["legs"]]
                end_long = [leg["end_location"]["lng"] for leg in routes["legs"]]
                distances = [round(leg["distance"]["value"]/1000, 3) for leg in routes["legs"]]
                travel_times = [round(leg["duration"]["value"] / 60) for leg in routes["legs"]]
                optimized_order = directions['routes'][0]['waypoint_order']
                optimized_waypoints = [self.__locations[i] for i in optimized_order]
                start_locations = [origin] + optimized_waypoints
                end_locations = optimized_waypoints + [destination]
                urls = []
                travel_modes = []
                for i in range(0, len(travel_times)):
                    time = travel_times[i]
                    start = start_locations[i]
                    end = end_locations[i]
                    url, travel_mode = self._get_url(start, end, time)
                    urls.append(url)
                    travel_modes.append(travel_mode)
                print('Optimized route:')
                print(f'Start at: {origin}')
                for i, waypoint in enumerate(optimized_waypoints):
                    print(f'Stop {i + 1}: {waypoint}')
                print(f'End at: {destination}')
            else:
                print(f"Error in API response: {directions['status']}")
        else:
            print(f"Error in HTTP request: {response.status_code}")
        route_dict = {"Start Address": start_locations, "Start Latitude": start_lat, "Start Longitude": start_long, "End Address": end_locations, "End Latitude": end_lat, "End Longitude": end_long, "Travel distance (km)": distances, "Travel time (min)": travel_times, "Travel mode": travel_modes, "Google Maps Link": urls}
        self.__route_df = pd.DataFrame(route_dict)
        self.__is_optimised = True
        return self.__route_df
        
    def _get_url(self, start, end, time):
        start = start.replace(',', '%2C')
        start = start.replace(' ', '+')
        end = end.replace(',', '%2C')
        end = end.replace(' ', '+')
        travel_mode = 'walking' if time <= self.__MAX_WALKING_TIME else 'public transport'
        base = "https://www.google.com/maps/dir/?api=1"
        query = f'&origin={start}&destination={end}&travelmode={travel_mode}'
        return base + query, travel_mode
    
    def get_route(self):
        if not self.__route:
            print('Error: Route has not been generated')
        return self.__route
    
    def generate_pdf(self):
        df = self.__route_df
        selected_fields = ['Start Address', 'End Address', "Travel mode", "Google Maps Link"]
        df = df[selected_fields]
        df = df.map(lambda s: s.title() if type(s) == str else s)
        # Reset index to start from 1
        df.index = df.index + 1
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Index'}, inplace=True)
        cluster_num = self.__cluster_num
        def create_hyperlink_paragraph(url, text):
            styleSheet = getSampleStyleSheet()
            hyperlink_style = styleSheet['BodyText']
            hyperlink_style.textColor = colors.blue
            hyperlink_style.underline = True
            hyperlink = Paragraph(f'<a href="{url}">{text}</a>', hyperlink_style)
            return hyperlink
        # Create a PDF document
        pdf_file = f"pdfs/Survey Locations for Cluster {cluster_num + 1:02}.pdf"
        pdf = SimpleDocTemplate(pdf_file, pagesize=landscape(A4))
        
        # Convert DataFrame to list of lists
        data_list = [df.columns.values.tolist()] + df.astype('str').values.tolist()
        for row in data_list[1:]:
            row[-1] = create_hyperlink_paragraph(row[-1].lower(), "link")
        
        styleSheet = getSampleStyleSheet()
        default_style = styleSheet['BodyText']
        col_widths = []
        for col_idx in range(len(data_list[0])):
            max_width = max(stringWidth(str(data_list[row_idx][col_idx]), default_style.fontName, default_style.fontSize) for row_idx in range(len(data_list)))
            if col_idx == len(data_list[0]) - 1:
                col_widths.append(80)
            else:
                col_widths.append(max_width + 10)  # Add some padding

        # Create a Table with the data and column widths
        table = Table(data_list, colWidths=col_widths)

        # Add style to the table
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ])
        table.setStyle(style)

        title_style = ParagraphStyle('Title', fontName='Helvetica-Bold',
                                     fontSize=16, spaceAfter=20, alignment=1)

        # Build the PDF
        title_paragraph = Paragraph(f"Cluster {cluster_num + 1}", title_style)
        elements = [title_paragraph, table]
        pdf.build(elements)

        print(f'PDF saved as {pdf_file}')

    def __str__(self) -> str:
        return f"Cluster: {self.__cluster_num}\nNumber of Locations: {self.__num_locations}\
                \nMaximum walking threshold: {self.__MAX_WALKING_TIME} minutes\
                \nSurveyor: {self.__surveyor}\nStart Address: {self.__start_address}\
                \nEnd Address: {self.__end_address}\nHas Optimised Route: {self.__is_optimised}"

    def save_route_df(self):
        df = self.__route_df
        cluster_num_column = [self.__cluster_num + 1] * len(df)
        df['cluster'] = cluster_num_column
        df = df.drop('Google Maps Link', axis=1)
        df.to_excel(f'./clustercsv/cluster{self.__cluster_num+1}.xlsx')
        return df
    