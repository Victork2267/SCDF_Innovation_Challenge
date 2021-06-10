import streamlit as st

st.sidebar.header("FireOn")
st.sidebar.subheader("View CCTV Cluster")
add_button_cluster1 = st.sidebar.button("View Cluster 1")
add_button_cluster2 = st.sidebar.button("View Cluster 2")
add_button_cluster3 = st.sidebar.button("View Cluster 3")
add_button_cluster4 = st.sidebar.button("View Cluster 4")
add_button_cluster5 = st.sidebar.button("View Cluster 5")

st.title("CCTV Alert Screen")
#st.success("All Green")
st.error("Fire detected in Cluster 1 CCTV cameras")
st.warning("Object detected as possible source of Fire in Cluster 1")
video_file = open("test1_output.mp4", "rb")
video_bytes = video_file.read()
st.video(video_bytes)

if add_button_cluster1:
    st.title("Cluster 1 CCTV cameras")
    st.subheader("CCTV1")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV2")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV3")
    st.video(data, format='video/mp4', start_time=0)

if add_button_cluster2:
    st.title("Cluster 2 CCTV cameras")
    st.subheader("CCTV1")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV2")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV3")
    st.video(data, format='video/mp4', start_time=0)

if add_button_cluster3:
    st.title("Cluster 3 CCTV cameras")
    st.subheader("CCTV1")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV2")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV3")
    st.video(data, format='video/mp4', start_time=0)

if add_button_cluster4:
    st.title("Cluster 4 CCTV cameras")
    st.subheader("CCTV1")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV2")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV3")
    st.video(data, format='video/mp4', start_time=0)

if add_button_cluster5:
    st.title("Cluster 5 CCTV cameras")
    st.subheader("CCTV1")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV2")
    st.video(data, format='video/mp4', start_time=0)
    st.subheader("CCTV3")
    st.video(data, format='video/mp4', start_time=0)
