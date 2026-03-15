from pinterest_dl import PinterestDL

# Search for images
images = PinterestDL.with_api().search_and_download(
    query="memes",
    output_dir="test_memes",
    num=30
)