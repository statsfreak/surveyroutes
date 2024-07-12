import streamlit as st
from v2_deploy import Assignment
from streamlit.components.v1 import html
from streamlit_folium import st_folium
import os

# st.set_page_config(layout="wide")

@st.cache_resource
def create_assignment(max_cluster_size, max_walking_time, max_cluster_distance):
    assignment = Assignment(max_cluster_size, max_walking_time, max_cluster_distance)
    return assignment

@st.experimental_fragment
def show_download_button(filename):
    with open(filename, "rb") as file:
        st.download_button(label=f"Download Directions for all clusters",
            data=file,
            file_name=file_name,
            mime="application/pdf")


if "first_form" not in st.session_state:
    st.session_state["first_form"] = None

if "surveyor_form" not in st.session_state:
    st.session_state['surveyor_form'] = None

# Title of the Streamlit app
st.title("Survey Locations Clustering and Route Optimisation Tool")

# Slider input fields
submission_form = st.form(key='submission')
submission_form.subheader("Cluster Configuration and File Upload")
max_cluster_size = submission_form.slider("Maximum Cluster Size", min_value=1, max_value=20, value=12, step=1)
max_walking_time = submission_form.slider("Maximum Walking Time (in minutes)", min_value=5.0, max_value=60.0, value=20.0, step=0.5)
max_cluster_distance = submission_form.slider("Maximum distance between cases within each cluster (in km)", min_value=1.0, max_value=15.0, value=7.0, step=0.5)

# File uploader for CSV or Excel files
uploaded_file = submission_form.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
submission_form.write("Note: File should have block number, street name, storey, unit number and postal code columns")

submit_button = submission_form.form_submit_button('Submit')

# Submit button
if submit_button:
    st.session_state['first_form'] = True
    st.session_state['surveyor_form'] = False
    folder_path = './cluster_maps'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    folder_path = './pdfs'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    

if st.session_state['first_form']:
    assignment = create_assignment(max_cluster_size, max_walking_time, max_cluster_distance)
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == 'csv':
            st.write("You have uploaded a CSV file.")
            assignment.upload_csv(uploaded_file)
        elif file_extension in ["xlsx", "xls"]:
            st.write("You have uploaded an Excel file.")
            assignment.upload_excel(uploaded_file)
        else:
            st.write("Please upload a valid CSV or Excel file.")
    assignment.geocode_dataset()
    assignment.cluster_dataset()
    st.write(f'There are {assignment.get_num_locations()} locations in {assignment.get_num_clusters()} clusters.')
    cluster_map = assignment.plot_map_all_cluster()
    with open ('./cluster_maps/cluster_combined.html', 'r', encoding='utf-8') as f:
        folium_map = f.read()
    html(folium_map, height=400)
    clusters = assignment.get_clusters()
    surveyor_names = []
    start_addresses = []
    end_addresses = []
    api_key = None
    with st.form('surveyor form'):
        st.title("Enter surveyor information")
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
        st.subheader(f'Google Cloud API Key')
        api_key = st.text_input('Enter Google Cloud API Key', type="password")
        surveyor_submit = st.form_submit_button("Submit surveyors' info")
    if surveyor_submit:
        st.session_state['surveyor_form'] = True
        st.write('Submitted surveyor details')
        assignment.assign_all_surveyors(surveyor_names, start_addresses, end_addresses)
        assignment.optimise_all_routes(api_key)
        assignment.generate_all_pdfs_and_maps()
        assignment.generate_combined_pdf()
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