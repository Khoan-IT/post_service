import os
import sys
import json
import copy
import time

import grpc
import intent_slot_service_pb2
import intent_slot_service_pb2_grpc

from underthesea import word_tokenize
from normalizer import V2PostNormalizer
from result.elastic_formatter import PostFormatter

MONITOR_SERVER_INTERFACE = os.environ.get('HOST', 'localhost')
MONITOR_SERVER_PORT = int(os.environ.get('PORT', 5002))

CHANNEL_IP = f"{MONITOR_SERVER_INTERFACE}:{MONITOR_SERVER_PORT}"

def main():
    channel = grpc.insecure_channel(CHANNEL_IP)
    stub = intent_slot_service_pb2_grpc.ISServiceStub(channel)
    normalizer = V2PostNormalizer()
    # message = """
    #   [CÔNG TÁC XÃ HỘI]
    #     Thân chào tất cả các bạn sinh viên,
    #     Đội Xung Kích xin gửi form đăng kí CTXH hỗ trợ quán cơm, thông tin cụ thể như sau:
    #     --------------------------------------------------------
    #     HOẠT ĐỘNG: HOẠT ĐỘNG HỖ TRỢ QUÁN CƠM.
    #     - Thời gian: từ ngày 11 đến ngày 12 tháng 11 năm 2022
    #     - Địa điểm: 
    #             + ĐỊA CHỈ QUÁN CƠM 2000: Số 14/1 Ngô Quyền, Hồ Chí Minh.
    #             + ĐỊA CHỈ QUÁN CƠM NỤ CƯỜI 7: 68/12 Lữ Gia, Quận 11, Hồ Chí Minh.
    #     - Số lượng: 05- 10 sinh viên / buổi.
    #     - Quyền lợi: 01 ngày CTXH.
    #     - Link đăng kí: https://forms.gle/gNPXWJwZREbZ2vue7
    #     --------------------------------------------------------
    #     * CHÚ Ý:
    #     - HOẠT ĐỘNG NÀY BẮT BUỘC CÁC BẠN MANG THEO THẺ SINH VIÊN (HOẶC GIẤY TỜ TÙY THÂN NHƯ CMND, BẰNG LÁI XE...) ĐỂ THUẬN TIỆN TRONG CÁC CÔNG TÁC ĐIỂM DANH.
    #     - Các bạn sau khi tham gia hoạt động nhớ chú ý CHỈ ĐIỂM DANH VÀO SỔ CỦA ĐỘI XUNG KÍCH KHOA ĐIỆN và KHÔNG ĐIỂM DANH VÀO BẤT KÌ QUYỂN SỔ NÀO KHÁC. Bất kì trường hợp nào điểm danh vào sổ khác chúng mình sẽ KHÔNG GIẢI QUYẾT.
    #     - Đây là những hoạt động tuyển rất ít sinh viên nên khi các bạn đăng ký tham gia thì phải đảm bảo là mình có mặt đầy đủ và đúng giờ. Mọi sai sót sẽ khiến việc hỗ trợ các quán cơm gặp khó khăn! Những bạn đến quá trễ hoặc về sớm khi chưa có sự cho phép của bên quản lý quán cơm sẽ KHÔNG ĐƯỢC tính ngày CTXH.
    #     - Các bạn đăng ký mà không tham gia sẽ bị thêm vào danh sách hạn chế để đảm bảo quyền lợi cho các bạn khác!
    #     Cảm ơn các bạn đã đọc tin!
    # """
    # message = " ".join(normalizer.v2_normalize(message))
    # print(message)
    # length_post = len(message.split())
    # print(length_post)
    # result = stub.IntentSlotRecognize(intent_slot_service_pb2.IntentSlotRecognizeRequest(message=message))
    # slots = json.loads(result.message)
    # for k, v in slots['slot'].items():
    #     print("{}: {}".format(k, v))
    es_formatter = PostFormatter()
    with open('./result/original/v2_raw_post_model.json', 'r') as f:
        posts = json.load(f)
        message = []
        for post in posts:
            temp_post = {"id": post["_id"]["$oid"]}
            temp_post['content'] = " ".join(normalizer.v2_normalize(post['content']))
            message.append(temp_post)
        message = json.dumps(message)
        result = stub.IntentSlotRecognize(intent_slot_service_pb2.IntentSlotRecognizeRequest(message=message))
        print(es_formatter.get_activities(json.loads(result.message)))
        
        # for index, post in enumerate(posts):
        #     message = " ".join(normalizer.v2_normalize(post['content']))
        #     length_post = len(message.split())
        #     result = stub.IntentSlotRecognize(intent_slot_service_pb2.IntentSlotRecognizeRequest(message=message))
        #     slots = json.loads(result.message)
        #     results[index].update({'length_post': length_post, 'result': slots['slot'], 'normalized_content': message})
        #     # print(message)
        #     # print(slots['slot'])
        #     # break
        # save_file = open('./result/slot_result/result_orig_gpt.json', 'w', encoding='utf-8')
        # json.dump(results, save_file, ensure_ascii=False)
        # save_file.close()
    
if __name__ == "__main__":
    main()
    # import nltk
    # nltk.download('punkt')