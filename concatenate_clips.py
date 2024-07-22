# from moviepy.editor import VideoFileClip, concatenate_videoclips

# def concat_clips(num_clips, clip_names):
#     # list1 = [VideoFileClip("prediction/video2_1.mp4"), VideoFileClip("prediction/video2_2.mp4"), VideoFileClip("prediction/video2_3.mp4"), VideoFileClip("prediction/video2_4.mp4"), VideoFileClip("prediction/video2_5.mp4")]
#     # list2 = [VideoFileClip("video2_1.mp4"), VideoFileClip("video2_2.mp4"), VideoFileClip("video2_3.mp4"), VideoFileClip("video2_4.mp4"), VideoFileClip("video2_5.mp4")]
#     list_mod = []
#     list_og = []
#     for i in clip_names:
#         try:
#             clip = VideoFileClip(i)
#             if clip.reader.nframes > 0:  # Check if the clip has non-zero frames
#                 list_mod.append(clip)
#                 print(f"Appended: {i}")
#             else:
#                 print(f"Skipped: {i} (no frames)")
#         except Exception as e:
#             print(f"Error processing {i}: {e}")
#         # print()
#         # list_og.append(VideoFileClip(i[11:]))

#     mod_clip = concatenate_videoclips(list_mod)
#     # og_clip = concatenate_videoclips(list_og)

#     mod_clip.write_videofile("highlights.mp4")
#     # og_clip.write_videofile("original.mp4")

#     pass

# from moviepy.editor import VideoFileClip, concatenate_videoclips
# import psutil
# import os

# def concat_clips(num_clips, clip_names):
#     list_mod = []
    
#     for i in clip_names:
#         try:
#             clip = VideoFileClip(i)
#             if clip.reader.nframes > 0:  # Check if the clip has non-zero frames
#                 list_mod.append(clip)
#                 print(f"Appended: {i}")
#             else:
#                 print(f"Skipped: {i} (no frames)")
#         except Exception as e:
#             print(f"Error processing {i}: {e}")
        
#         # Monitor memory usage
#         process = psutil.Process(os.getpid())
#         print(f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

#     if list_mod:
#         try:
#             mod_clip = concatenate_videoclips(list_mod)
#             mod_clip.write_videofile("highlights.mp4")
#         except Exception as e:
#             print(f"Error concatenating clips: {e}")
#     else:
#         print("No valid clips to concatenate")


from moviepy.editor import VideoFileClip, concatenate_videoclips

def concat_clips(num_clips, clip_names):
    # list1 = [VideoFileClip("prediction/video2_1.mp4"), VideoFileClip("prediction/video2_2.mp4"), VideoFileClip("prediction/video2_3.mp4"), VideoFileClip("prediction/video2_4.mp4"), VideoFileClip("prediction/video2_5.mp4")]
    # list2 = [VideoFileClip("video2_1.mp4"), VideoFileClip("video2_2.mp4"), VideoFileClip("video2_3.mp4"), VideoFileClip("video2_4.mp4"), VideoFileClip("video2_5.mp4")]

    list_mod = []
    list_og = []
    for i in clip_names:
        try:
            print("Name of clip", i)
            list_mod.append(VideoFileClip(i))
            # list_og.append(VideoFileClip(i[11:]))
        except KeyError as e:
            print(f"Error processing {i}: {e}")

    mod_clip = concatenate_videoclips(list_mod)
    # og_clip = concatenate_videoclips(list_og)

    mod_clip.write_videofile("highlights.mp4")
    # og_clip.write_videofile("original.mp4")



#concat_clips(8, ["prediction/original_short_1.mp4","prediction/original_short_2.mp4","prediction/original_short_3.mp4","prediction/original_short_4.mp4","prediction/original_short_5.mp4","prediction/original_short_6.mp4","prediction/original_short_7.mp4","prediction/original_short_8.mp4"])