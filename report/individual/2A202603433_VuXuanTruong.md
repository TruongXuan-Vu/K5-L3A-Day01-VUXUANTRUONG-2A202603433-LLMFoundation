# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603433
- **Họ tên:** Vũ Xuân Trường
- **SEED** (in ở ô đầu notebook): **1072** · thử thách cá nhân: đề **#5** (Embedding)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ (chạy đến hết Phần 3; các ô gọi API LLM ở Phần 4–5 chưa có output vì thiếu key khi chạy lại — xem mục 6, 7) |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (nguồn: ô "Báo cáo cuối" và `lab1_report.json`, SEED 1072):

| Model | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.880 | 0.857 | 0.40 | 0.00 |
| 2. BoW + MLP (deep learning) | 0.939 | 0.974 | 4.52 | 0.00 |
| 3a. Encoder (MiniLM) + kNN | 0.887 | 0.950 | 136.88 | 0.00 |
| 3b. Encoder (MiniLM) + LogReg | **0.957** | 0.950 | 41.73 | 0.00 |
| 3c. Encoder (MiniLM) + MLP head | 0.929 | 0.974 | 62.52 | 0.00 |

Bổ sung: precision/recall của model 1 là 0.986 / 0.793 (recall thấp nhất), model 3b cân bằng 0.957 / 0.957. Tính tay trên giấy (xem `hand_calc.jpg`): baseline "luôn đoán ham" = 4516/5169 = **0.87** — mọi model đều vượt baseline nhưng model 1 chỉ hơn ~0.10 accuracy.

## 3. Dự đoán và thực tế

**Dự đoán 1 — `dl_relu` (Thí nghiệm 2.B, bỏ ReLU).** Mình dự đoán: hai lớp `Linear` không có ReLU thì gộp thành một phép biến đổi tuyến tính, tương đương logistic regression Phần 1, nên F1 sẽ rơi về mức 0.88, không hơn MLP. Thực tế: **F1 = 0.939 — ngang bằng MLP có ReLU**, cao hơn Phần 1 tới 0.06. Lệch vì mình nhầm giữa *khả năng biểu diễn* và *nghiệm tìm được*: bỏ ReLU đúng là mất phi tuyến, nhưng mạng vẫn có 128k tham số và được tối ưu bằng Adam theo mini-batch, nên tìm được nghiệm tốt hơn nghiệm mà gradient descent full-batch tự viết ở Phần 1 tìm được. Bài học: hai mô hình cùng lớp hàm expressive không có nghĩa là cùng kết quả — cách tối ưu cũng quyết định.

**Dự đoán 2 — `emb_tricky` (Thí nghiệm 3.B, 6 tin ngụy trang).** Mình dự đoán: BoW đúng ~3/6 (bị "free/won/call" trong tin ham đánh lừa, bỏ lọt spam không có từ khóa), Encoder+LogReg đúng 5–6/6. Thực tế: **BoW đúng 5/6, Enc+LR cũng 5/6** — hai bên bằng nhau, và cùng sai đúng một tin ("Last chance! Your account rewards expire tonight…", thật = spam). Lệch vì bộ 6 tin của lab thật ra rất "đơn giản" cho BoW: các tin ham chỉ có 1 tin chứa từ khóa mạnh, còn tin spam vẫn đủ từ dễ nhận (complimentary, chosen, at no cost). Bộ case khó thật sự nằm ở stress test 50 case (mục 4), nơi BoW tụt mạnh.

**Dự đoán 3 — `emb_search` (Thí nghiệm 3.A, tìm theo nghĩa).** Mình dự đoán kết quả gần nhất là spam (đúng) nhưng BoW "chỉ khớp được vài từ chung chung như you, to, now". Thực tế: query có **12/13 từ nằm trong vocab BoW** — gần như khớp trọn câu, không phải "vài từ". Lệch ở chi tiết nhưng thay đổi cách hiểu: BoW không thiếu từ để khớp, nó thiếu khả năng *xếp hạng theo nghĩa* — embedding chỉ cần 6 hàng xóm gần nhất đều đúng spam (0.546, 0.544…), trong khi độ tương tự đếm từ không phân biệt "reply now to claim" (spam) với "reply now" trong hội thoại thường.

## 4. Phân tích 50 case khó

Số liệu từ lần chạy stress test của mình (đã ghi ở ô Câu hỏi 6 trong notebook):

| Model | F1 test thường (SUB) | F1 50 case khó | Tụt |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.857 | 0.286 | **−0.571** |
| 2. BoW + MLP | 0.974 | 0.500 | −0.474 |
| 3c. Emb + MLP head | 0.974 | 0.533 | −0.441 |
| 3b. Emb + LogReg | 0.950 | 0.553 | −0.397 |
| 3a. Emb + kNN | 0.950 | **0.750** | −0.200 |

**Model tụt nhiều nhất: BoW + LogReg (−0.571)**; kNN tụt ít nhất. Điểm test thường quá lạc quan vì tin spam trong test lặp lại cùng kiểu từ khóa (free, win, prize) có trong train, nên chỉ cần nhận diện từ khóa quen là được; 50 case khó thì cố tình bỏ những tín hiệu đó, lại chỉ có 40 tin ở SUB nên vài sai lệch đã làm F1 rơi mạnh. Accuracy 0.50–0.60 của các model BoW và encoder còn thấp hơn cả mức "đoán tất cả là spam" (29/50 = 58%) — gần như không hơn đoán mò.

**Nhóm case khó nhất:** với BoW là *spam giọng tự nhiên* (BoW+LogReg đúng 0%) và *né từ khoá / thao túng* (đúng 12–17%); với embedding là *natural_spam* (10%) và *manipulation* (25%).

**Case cụ thể 1 — natural_spam (#14):** *"Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back"* (thật = spam). BoW sai vì tin không có một từ khóa spam nào: toàn bộ từ đều là từ hội thoại đời thường, trọng số cộng dồn nghiêng về ham. Cơ chế: BoW nhìn văn bản như một túi token rời rạc — khi tín hiệu nằm ở *giọng điệu và mục đích của cả câu* thay vì từng từ, nó mù hẳn.

**Case cụ thể 2 — manipulation (#21):** *"Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account?"* (thật = spam). Embedding sai vì về **nghĩa**, tin này giống hệt tin nhắn gia đình thật: người thân nhắn số mới, xin tiền gấp — vector cả câu nằm sát vùng ham. Cơ chế: lừa "Hi Mum" khai thác đúng điểm mạnh của embedding (hiểu ngữ cảnh tin nhắn hợp lệ) và dùng ngược nó; đây là nhóm mà cả embedding lẫn LLM đều khó hơn BoW ít.

## 5. Thử thách cá nhân

**Đề #5 theo SEED 1072:** *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*

Cách làm: viết 4 tin vào ô thử thách trong notebook (code đã ở ô `mine = [...]`), chạy so sánh p_BoW (từ `predict_proba` Phần 1) với p_Enc+LR (encoder MiniLM + LogReg 3b), kèm đếm số từ của mỗi tin nằm trong vocab BoW:

1. *"Cl@im ur c@sh r3ward n0w, tap the l1nk b4 it expires"* — spam né từ khoá kiểu leetspeak: hầu hết token (c@sh, r3ward, l1nk, b4) không có trong vocab nên BoW gần như không có tín hiệu spam để cộng; embedding vẫn thấy cả câu mang nghĩa "thúc giục nhận thưởng".
2. *"You have been selected for a complimentary cruise, reply YES to confirm"* — spam né từ khóa free/win: "selected", "complimentary", "reply YES" là từ thường, nhưng cấu trúc "bạn được chọn + xác nhận để nhận" là mẫu câu trúng thưởng; embedding đặt câu này gần cụm spam đã thấy ở 3.A, BoW chỉ còn tín hiệu yếu.
3. *"Are you free tonight? Let's grab dinner after work"* — ham chứa "free": từ "free" có trọng số đẩy spam mạnh, nên đây là bẫy của BoW; tuy nhiên tương tự tin "Hey are you free for lunch tomorrow?" ở 3.B mà BoW đoán đúng, nên khả năng BoW vẫn qua được nhờ các từ đời thường còn lại kéo ngược.
4. *"Did you win your tennis match? Call me when you get home"* — ham chứa "win" + "call": giống tin "I won the match yesterday, call me when you can" ở 3.B (BoW đoán đúng) — "win" ở đây là động từ thi đấu, và tin có nhiều từ ham mạnh (you, me, when, home).

Kết quả kỳ vọng có cơ chế (đối chiếu trực tiếp với Thí nghiệm 3.B, cùng mô hình cùng seed): BoW sai tin 1 (mất token do obfuscation) và rủi ro cao ở tin 3, đúng khoảng 2–3/4; Enc+LR phân loại đúng cả 4 vì nhìn nghĩa của cả câu. Giải thích: hai nhóm bẫy đánh vào đúng hai điểm yếu khác nhau — obfuscation đánh vào *vocab tĩnh* của BoW, còn "free/won" trong hội thoại đánh vào *việc BoW cộng trọng số từng từ độc lập, không thấy ngữ cảnh*; embedding chỉ yếu khi cả câu giả danh tin hợp lệ (nhóm manipulation ở mục 4). Kết quả chạy thật của 4 tin này nằm ở ô thử thách trong notebook.

## 6. Phương án cho sản phẩm thật

Yêu cầu: lọc 10 triệu SMS/ngày (~116 tin/giây trung bình), **p95 < 50 ms/tin**.

**Chọn: 3b — Encoder (MiniLM) + LogReg**, đây là phương án duy nhất đạt đồng thời cả hai ràng buộc:
- **Độ trễ:** p95 = 41.73 ms/tin < 50 ms (kNN 136.88 ms, 3c 62.52 ms và LLM — hàng trăm ms mỗi lần gọi API — đều vượt SLA).
- **Chất lượng:** F1 test 0.957, cân bằng precision/recall 0.957/0.957, F1 SUB 0.950 — tốt nhất nhóm đạt SLA.
- **Chi phí biến số:** USD/1k tin = 0 (chạy tại chỗ); encoder MiniLM chỉ ~90 MB nên hạ tầng phục vụ rẻ.

Dẫn số liệu chấp nhận đánh đổi:
1. **p95 41.7 ms sát ngưỡng 50 ms** — traffic đột biến (vụ spam 10×) sẽ vượt SLA. Phương án dự phòng: router 2 tầng — BoW+MLP (p95 4.52 ms, F1 SUB 0.974 cao nhất bảng) xử lý các tin có xác suất chắc chắn, chỉ đẩy tin lưỡng lự sang encoder.
2. **Chi phí encode khi train:** 112.7s cho 3.619 tin (~32 tin/s) — train lại model định kỳ tốn tài nguyên nhiều hơn hẳn BoW (8.2s), nhưng inference ổn định vì encoder đóng băng.
3. **Recall 0.957** nghĩa là ~4% spam lọt; với 10 triệu tin/ngày, cần lớp giám sát + cập nhật dữ liệu train thường xuyên, vì 50 case khó đã cho thấy spam giọng tự nhiên là điểm yếu của cả embedding (đúng 10% ở natural_spam).
4. **Không dùng LLM làm lớp chính:** ngoài độ trễ, tính chi phí từ bài tính tay (gpt-6-luna, 100 token input/1 token output, 1 triệu tin/ngày = $10.50/ngày) → 10 triệu tin/ngày ≈ **$105/ngày ≈ $3.150/tháng**, gấp nhiều lần chi phí hạ tầng chạy 3b.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Sau lab | Đọc file báo cáo mẫu, liệt kê những chỗ còn thiếu của mình | Chỉ ra SEED trống, 7 mục chưa điền, và các ô Phần 4–5 trong notebook chưa có output (thiếu key khi chạy lại) | Tự mở lại notebook đối chiếu từng ô — đúng là các ô gọi API chưa chạy, các ô khác còn nguyên output |
| Viết report | Nhờ AI tổng hợp bảng số liệu và nháp phân tích 7 mục từ notebook + `lab1_report.json` | Chép bảng so sánh (F1, F1 SUB, p95, USD), 10 tin sai ở Phần 1, số liệu 50 case khó đã ghi trong notebook, giá trị tính tay từ `hand_calc.jpg` | Tự kiểm tra từng con số với `lab1_report.json` và ô "Báo cáo cuối"; phần 3 dự đoán lệch nhất do mình tự chọn và tự đánh giá lại lý do; phần thử thách cá nhân ghi rõ kết quả là suy luận có cơ chế từ Thí nghiệm 3.B vì ô chạy cuối chưa in output |

Ba điều mình hiểu ra hôm nay mà trước đây chưa hiểu:

1. **Accuracy là con số gây hiểu lầm nhất với dữ liệu lệch.** Đoán "tất cả là ham" đã được 0.87 (tự tính tay), nên con số 97–99% của các model gần như không nói gì — phải nhìn F1, và nhìn đúng tập: điểm test thường 0.98 vẫn có thể rơi xuống 0.29 trên 50 case khó.
2. **Model "tương đương" về lý thuyết không cho kết quả như nhau khi cách tối ưu khác** (bỏ ReLU vẫn đạt 0.939 nhờ Adam + mini-batch) — và ngược lại, model tốt ở test không tự động dùng được: 3a F1 đẹp nhưng p95 136 ms là loại ngay khỏi sản phẩm.
3. **Mỗi thế hệ AI hỏng theo một cách khác nhau:** BoW hỏng ở từ vựng (obfuscation, giọng tự nhiên), embedding hỏng khi cả câu giả danh tin hợp lệ (lừa "Hi Mum"), LLM hỏng khi bị chỉ đạo qua chính dữ liệu (prompt injection). Chọn phương án sản phẩm là chọn cách hỏng nào mình chịu được, chứ không phải tìm model "tốt nhất" trừu tượng.
