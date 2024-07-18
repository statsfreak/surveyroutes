import streamlit as st
from v2_deploy import Assignment
from streamlit.components.v1 import html
from streamlit_folium import st_folium
import os
import pandas as pd
import time

# function to create the assignment object (for the day's route assignment)
def create_assignment(max_cluster_size, max_walking_time, max_intra_cluster_distance):
    assignment = Assignment(max_cluster_size, max_walking_time, max_intra_cluster_distance)
    return assignment

# Create download button for the pdf sheet
@st.experimental_fragment
def show_download_button(filename):
    with open(filename, "rb") as file:
        st.download_button(label=f"Download Directions for all clusters",
            data=file,
            file_name=filename,
            mime="application/pdf")

@st.experimental_fragment
def show_locations_excel_template(input_data, filename, button_label):
    df = pd.DataFrame(input_data)
    df.to_excel(filename, index=False)
    with open(filename, "rb") as file:
        st.download_button(label=button_label,
                       data = file,
                       file_name = filename,
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

### Initialise the clustering parameters form and the surveyor form in the session state ###
if "clustering_parameters" not in st.session_state:
    st.session_state["clustering_parameters"] = None
if "surveyor_form" not in st.session_state:
    st.session_state['surveyor_form'] = None

### Title of the Streamlit app ###
st.title("Survey Locations Clustering and Route Optimisation Tool")

# Locations file template
st.text('Locations Data Template')
locations_template = {'Block':[], 'Street name': [], 'Floor': [], 'Unit': [],	'Postal Code': []}
locations_filename = './templates/locations.xlsx'
locations_button_label = 'Download Template for Locations Data'
show_locations_excel_template(locations_template, locations_filename, locations_button_label)

### Create form for the clustering parameters ###
submission_form = st.form(key='submission')
submission_form.subheader("Cluster Configuration and File Upload")
# Max cluster size
max_cluster_size = submission_form.slider("Maximum Cluster Size", min_value=1, max_value=20, value=12, step=1)
# Max walking time
max_walking_time = submission_form.slider("Maximum Walking Time (in minutes)", min_value=5.0, max_value=60.0, value=20.0, step=0.5)
# Max intra_cluster distance
max_intra_cluster_distance = submission_form.slider("Maximum distance between cases within each cluster (in km)", min_value=1.0, max_value=15.0, value=7.0, step=0.5)
# Locations file uploader
allowed_extensions = ['csv', 'xlsx']
uploaded_file = submission_form.file_uploader("Upload a CSV or Excel file", type=allowed_extensions)
submission_form.write("Note: File should have block number, street name, storey, unit number and postal code columns")
# Submit button for the clustering parameters form
submit_button = submission_form.form_submit_button('Submit')

### Clear relevant files and folders upon submission ###
if submit_button:
    # Establish the state of the clustering parameters and surveyor form
    st.session_state['clustering_parameters'] = True
    st.session_state['surveyor_form'] = False
    # Clear folder with the cluster maps
    folder_path = './cluster_maps'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # Clear folder with the pdf file
    folder_path = './pdfs'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    ### Perform clustering of the locations ###
    with st.spinner('Clustering locations'):
        # create the assignment with the specified parameters
        assignment = create_assignment(max_cluster_size, max_walking_time, max_intra_cluster_distance)
        # Process the uploaded file based on its extension.
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension == 'csv':
                # st.write("You have uploaded a CSV file.")
                assignment.upload_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                # st.write("You have uploaded an Excel file.")
                assignment.upload_excel(uploaded_file)
        # Geocode the uploaded file
        try:
            assignment.geocode_dataset()
        except ValueError as e:
            st.error(e)
            st.stop()
        # Cluster the locations and output the folium map
        assignment.cluster_dataset()
        st.write(f'There are {assignment.get_num_locations()} locations in {assignment.get_num_clusters()} clusters.')
        cluster_map = assignment.plot_map_all_cluster()
    st.session_state['assignment'] = assignment


if st.session_state['clustering_parameters']:
    # Display the map showing the clustered locations
    with open ('./cluster_maps/cluster_combined.html', 'r', encoding='utf-8') as f:
        folium_map = f.read()
    html(folium_map, height=400)
    # Initialise the surveyor names, start and end addresses and api key
    assignment = st.session_state['assignment']
    clusters = assignment.get_clusters()
    surveyor_names = []
    start_addresses = []
    end_addresses = []
    api_key = None
    # Create form for users to submit the surveyor information
    with st.form('surveyor form'):
        st.title('Submit Google API Key and Surveyors Info')
        api_key = st.text_input('Enter Google Cloud API Key', type="password")
        st.write(f"Submit the name, start addresses and end addresses for {assignment.get_num_clusters()} surveyors.")
        st.write("Default address (18 Havelock Road) will be used if address information left blank.")
        # File upload option
        with st.expander('File Upload'):
            surveyor_file_upload = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
        # Manual input option
        with st.expander('Manual input'):
            st.write('Do not attach a file if using manual input as the file will override manual input')
            for i in range(assignment.get_num_clusters()):
                st.subheader(f'Cluster {i + 1}')
                name_label = f'Cluster {i + 1} surveyor name'
                name_val = st.text_input(name_label)
                start_address_label = f'Cluster {i + 1} surveyor start address'
                start_address_val = st.text_input(start_address_label)
                end_address_label = f'Cluster {i + 1} surveyor end address'
                end_address_val = st.text_input(end_address_label)
                surveyor_names.append(name_val)
                start_addresses.append(start_address_val)
                end_addresses.append(end_address_val)            
        surveyor_submit = st.form_submit_button("Submit surveyors' info")
    st.text('Surveyors\' Data Template')
    surveyors_template = {'Name':[], 'Start Address': [], 'End Address': []}
    surveyors_filename = './templates/surveyors.xlsx'
    surveyors_button_label = 'Download Template for Surveyors Data'
    show_locations_excel_template(surveyors_template, surveyors_filename, surveyors_button_label)
    # Upon submission of the surveyor information
    if surveyor_submit:
        # Check whether API key has been submission
        if api_key == '':
            st.error('Please enter the API Key')
            st.stop()
        st.session_state['surveyor_form'] = True
        st.write('Submitted surveyor details')
        # Process the file upload if a file was uploaded.
        if surveyor_file_upload is not None:
            file_extension = surveyor_file_upload.name.split(".")[-1]
            if file_extension == 'csv':
                df = pd.read_csv(surveyor_file_upload)
            elif file_extension == 'xlsx':
                df = pd.read_excel(surveyor_file_upload)
            # Check that file contains surveyor information for all clusters
            if len(df) != assignment.get_num_clusters():
                st.error(f'Please submit the surveyor information for {assignment.get_num_clusters()} surveyors')
                st.stop()
            df.columns = df.columns.str.lower()
            surveyor_names = list(df['name'])
            start_addresses = list(df['start address'])
            end_addresses = list(df['end address'])
        # Assign the surveyors and optimise the route for each cluster
        with st.spinner('Optimising routes'):
            assignment.assign_all_surveyors(surveyor_names, start_addresses, end_addresses)
            try:
                assignment.optimise_all_routes(api_key)
            except ValueError as e:
                st.error(e)
                st.stop()
            # Generate the pdfs and maps for each of the routes
            assignment.generate_all_pdfs_and_maps()
            # Generate the combined pdf for all the clusters
            assignment.generate_combined_pdf()
            # Display the combined pdf and the maps of the routes
            if st.session_state['surveyor_form']:
                file_name = f'./pdfs/combined_survey_locations.pdf'
                if os.path.exists(file_name):
                    show_download_button(file_name)
                else:
                    print(f"Error: File {file_name} does not exist")
                for i, route_df in enumerate(assignment.get_optimised_routes()):
                    st.write(f"Cluster {i + 1}")
                    with open (f'./cluster_maps/cluster_{i + 1}.html', 'r', encoding='utf-8') as f:
                        folium_map = f.read()
                    html(folium_map, height=400)
