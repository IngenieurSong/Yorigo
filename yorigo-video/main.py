#-*- coding:utf-8 -*-

import scene_detect
import ocr
import os
from pytube import YouTube
from scene_match_ocr import match_scene_steps, combine_scene_timestemp
from scene_match_clip import scene_explanation_matching

# 사용자가 업로드한 Youtube Url을 통해서 레시피 영상 다운로드
def download_video(video_url):
    download_folder = "./video"

    downloader = YouTube(video_url)
    stream = downloader.streams.get_highest_resolution()
    stream.download(download_folder, "video.mp4")
    print("download complete")

def delete_all_files_in_dir(file_path):
    if(os.path.exists(file_path)):
        for file in os.scandir(file_path):
            os.remove(file.path)
        return "remove all files"
    else:
        return "directory not found"

def main(recipe_steps, video_url, recipe_name, isCLIP = False):
    download_video(video_url)
    scene_time_list = scene_detect.save_scene_frame("./video/video.mp4", "./scene")

    if(isCLIP):
        step_result = scene_explanation_matching(recipe_steps, "./scene")
    else:
        sentences = ocr.detect_discription("./scene")

        step_result = match_scene_steps(recipe_steps, sentences, recipe_name)

    delete_all_files_in_dir("./scene")
    delete_all_files_in_dir("./cropped_scene")
    delete_all_files_in_dir("./video")

    combine_result = combine_scene_timestemp(step_result, scene_time_list)

    return combine_result

if(__name__ == "__main__"):
    download_video("https://www.youtube.com/watch?v=148MzAQHmsg")