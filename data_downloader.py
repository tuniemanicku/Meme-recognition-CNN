from pinterest_dl import PinterestDL

# Search for images
images = PinterestDL.with_api().search_and_download(
    query="funny memes",
    output_dir="memes",
    num=150
)