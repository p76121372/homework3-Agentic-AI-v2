import os
import subprocess
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, END

# 全域設定
MODEL_NAME = "gemini-2.5-flash-lite"

def Use_LLM(prompt_text, temperature=0.7):
    """
    統一的 LLM 調用函數
    
    Args:
        prompt_text (str): 要發送給 LLM 的文字內容
        temperature (float): LLM 的 temperature 參數，預設為 0.7
    
    Returns:
        str: LLM 的回應內容，如果失敗則返回空字串
    """
    try:
        import os
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        
        # 讀取 API Key
        with open('API_key.txt', 'r') as f:
            api_key = f.read().strip()
        os.environ['GOOGLE_API_KEY'] = api_key
        
        # 建立 LLM
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=temperature,
            max_retries=2,
            google_api_key=os.environ['GOOGLE_API_KEY'],
        )
        
        # 調用 LLM
        response = llm.invoke([HumanMessage(content=prompt_text)])
        return response.content
        
    except Exception as e:
        print(f"[!] LLM 調用失敗: {e}")
        return ""

def analyze_user_intent(user_input, current_num_rows, current_normal_prob, current_abnormal_prob, current_null_prob):
    """讓 LLM 分析用戶意圖，判斷是否要修改參數"""
    
    intent_prompt = f"""你是參數修改意圖分析專家。請分析用戶的輸入，判斷是否要修改參數。

當前參數狀態：
- num_rows: {current_num_rows}
- normal_prob: {current_normal_prob}
- abnormal_prob: {current_abnormal_prob}
- null_prob: {current_null_prob}

用戶輸入："{user_input}"

請分析用戶是否想要修改任何參數。如果是，請按照以下格式輸出：

[PARAM_UPDATE]
param_name=new_value
param_name=new_value
[/PARAM_UPDATE]

例如：
[PARAM_UPDATE]
num_rows=500
normal_prob=0.9
[/PARAM_UPDATE]

如果用戶只是詢問問題而不是要修改參數，請正常回答問題，不要包含 [PARAM_UPDATE] 標籤。

注意：
- num_rows 必須是整數
- normal_prob, abnormal_prob, null_prob 必須是 0-1 之間的小數
- 只有當用戶明確表示要修改參數時才輸出 [PARAM_UPDATE] 標籤"""

    # 使用統一的 LLM 函數
    return Use_LLM(intent_prompt, temperature=0.7)

def parse_llm_output(llm_response):
    """解析 LLM 輸出的參數修改指令"""
    import re
    
    # 查找 [PARAM_UPDATE] 標籤內的內容
    pattern = r'\[PARAM_UPDATE\](.*?)\[/PARAM_UPDATE\]'
    match = re.search(pattern, llm_response, re.DOTALL)
    
    if not match:
        return {}
    
    param_block = match.group(1).strip()
    param_updates = {}
    
    # 解析每一行的參數設定
    for line in param_block.split('\n'):
        line = line.strip()
        if '=' in line:
            try:
                param_name, value_str = line.split('=', 1)
                param_name = param_name.strip()
                value_str = value_str.strip()
                
                if param_name == 'num_rows':
                    param_updates[param_name] = int(float(value_str))
                elif param_name in ['normal_prob', 'abnormal_prob', 'null_prob']:
                    value = float(value_str)
                    if 0 <= value <= 1:
                        param_updates[param_name] = value
                    else:
                        print(f"[!] 參數 {param_name} 的值 {value} 不在有效範圍 [0,1] 內")
            except ValueError as e:
                print(f"[!] 無法解析參數行: {line}, 錯誤: {e}")
    
    return param_updates

# 狀態定義
class WorkflowState(TypedDict):
    num_rows: int
    normal_prob: float
    abnormal_prob: float
    null_prob: float
    file_exists: bool
    data_path: str
    data_generated: bool
    output: str
    anomaly_checked: bool
    # 新增對話相關狀態
    user_input: str
    llm_response: str
    params_confirmed: bool
    conversation_active: bool
    # 新增總結相關狀態
    summary_saved: bool
    llm_summary_generated: bool
    # 新增用戶查詢相關狀態
    user_question: str
    code_flag: str  # "yes" 或 "no"
    llm_code_response: str
    code_execution_result: str
    code_error: str
    query_completed: bool
    continue_asking: bool

def check_file(state: WorkflowState, config=None):
    data_path = os.path.join('utils', 'Data', 'testing.csv')
    exists = os.path.exists(data_path)
    state['file_exists'] = exists
    state['data_path'] = data_path
    state['output'] = data_path  # 設定 output 路徑
    return state

def prompt_params(state: WorkflowState, config=None):
    """初始化參數設定對話"""
    print("\n[!] utils/Data/testing.csv 不存在，啟動智能參數設定助手...")
    
    # 設定預設參數
    state['num_rows'] = 300
    state['normal_prob'] = 0.95
    state['abnormal_prob'] = 0.3
    state['null_prob'] = 0.05
    state['output'] = 'utils/Data/testing.csv'
    state['params_confirmed'] = False
    state['conversation_active'] = True
    
    print("=" * 80)
    print(f"""🤖 智能參數設定助手已啟動！
📋 當前預設參數：
   - 資料行數 (num_rows): {state['num_rows']}
   - 正常資料感測器正常機率 (normal_prob): {state['normal_prob']}
   - 異常資料感測器正常機率 (abnormal_prob): {state['abnormal_prob']}
   - 空值機率 (null_prob): {state['null_prob']}

💬 你可以：
   - 詢問任何關於參數的問題
   - 修改參數（例如："設定行數為500"）
   - 輸入 'yes' 開始用當前參數生成資料""")
    print("=" * 80)
    
    # 等待用戶輸入
    user_input = input("\n你: ").strip()
    state['user_input'] = user_input
    
    # 檢查是否確認參數
    if user_input.lower() in ['yes', 'y', '確定', '開始']:
        state['params_confirmed'] = True
        state['conversation_active'] = False
        print(f"\n✅ 確認使用參數生成資料！")
    
    return state

def user_intent_analysis(state: WorkflowState, config=None):
    """分析用戶意圖並處理參數修改"""
    
    try:
        # 分析用戶意圖
        intent_analysis = analyze_user_intent(
            state['user_input'], 
            state['num_rows'], 
            state['normal_prob'], 
            state['abnormal_prob'], 
            state['null_prob']
        )
        
        state['llm_response'] = intent_analysis
        
        # 解析參數修改指令
        param_updates = parse_llm_output(intent_analysis)
        if param_updates:
            print("=" * 80)
            for param_name, new_value in param_updates.items():
                if param_name == 'num_rows':
                    state['num_rows'] = new_value
                    print(f"✅ 已更新資料行數為: {new_value}")
                elif param_name == 'normal_prob':
                    state['normal_prob'] = new_value
                    print(f"✅ 已更新正常資料感測器正常機率為: {new_value}")
                elif param_name == 'abnormal_prob':
                    state['abnormal_prob'] = new_value
                    print(f"✅ 已更新異常資料感測器正常機率為: {new_value}")
                elif param_name == 'null_prob':
                    state['null_prob'] = new_value
                    print(f"✅ 已更新空值機率為: {new_value}")
            
            print(f"""
📋 更新後的參數：
   - 資料行數 (num_rows): {state['num_rows']}
   - 正常資料感測器正常機率 (normal_prob): {state['normal_prob']}
   - 異常資料感測器正常機率 (abnormal_prob): {state['abnormal_prob']}
   - 空值機率 (null_prob): {state['null_prob']}""")
            print("=" * 80)
        else:
            # 沒有參數修改，顯示 LLM 回應
            print("=" * 80)
            print(f"🤖 助手: {intent_analysis}")
            print("=" * 80)
        
        # 繼續對話，等待下一個輸入
        user_input = input("\n你: ").strip()
        state['user_input'] = user_input
        
        # 檢查是否確認參數
        if user_input.lower() in ['yes', 'y', '確定', '開始']:
            state['params_confirmed'] = True
            state['conversation_active'] = False
            print("=" * 80)
            print("✅ 確認使用參數生成資料！")
            print("=" * 80)
        
    except Exception as e:
        print(f"[!] LLM 分析失敗: {e}")
        # 如果 LLM 失敗，改用簡單模式
        state['conversation_active'] = False
        state['params_confirmed'] = True
        print("改用預設參數生成資料")
    
    return state

def simple_prompt_params(state: WorkflowState):
    """簡單版本的參數設定（當 LLM 不可用時）"""
    print("\n[!] 使用簡單模式設定參數")
    
    try:
        num_rows = input("生成行數 (num_rows, 預設300): ")
        num_rows = int(num_rows) if num_rows.strip() else 300
    except Exception:
        num_rows = 300
    try:
        normal_prob = input("normal label 正常值域機率 (normal_prob, 預設0.95): ")
        normal_prob = float(normal_prob) if normal_prob.strip() else 0.95
    except Exception:
        normal_prob = 0.95
    try:
        abnormal_prob = input("abnormal label 正常值域機率 (abnormal_prob, 預設0.3): ")
        abnormal_prob = float(abnormal_prob) if abnormal_prob.strip() else 0.3
    except Exception:
        abnormal_prob = 0.3
    try:
        null_prob = input("空值機率 (null_prob, 預設0.05): ")
        null_prob = float(null_prob) if null_prob.strip() else 0.05
    except Exception:
        null_prob = 0.05
    
    state['num_rows'] = num_rows
    state['output'] = 'utils/Data/testing.csv'
    state['normal_prob'] = normal_prob
    state['abnormal_prob'] = abnormal_prob
    state['null_prob'] = null_prob
    return state

def generate_data(state: WorkflowState, config=None):
    print("\n[Agentic Workflow] 執行 generate_data.py 產生測試資料...")
    cmd = [
        'python', 'utils/generate_data.py',
        '-n', str(state['num_rows']),
        '-o', state['output'],
        '--normal_prob', str(state['normal_prob']),
        '--abnormal_prob', str(state['abnormal_prob']),
        '--null_prob', str(state['null_prob'])
    ]
    subprocess.run(cmd)
    state['data_generated'] = True
    return state

def anomaly_detection(state: WorkflowState, config=None):
    print("\n[Agentic Workflow] 開始異常檢測...")
    try:
        # 動態導入 checking_agent
        sys.path.append('utils')
        from checking_agent import DataQualityAgent
        
        # 建立異常檢測代理
        agent = DataQualityAgent(state['output'])
        
        # 載入資料
        if not agent.load_data():
            print("[!] 異常檢測：資料載入失敗")
            state['anomaly_checked'] = False
            return state
        
        # 執行異常檢測（結合 rule-based 和 model-based）
        summary = agent.check_all_rows_combined()
        
        # 保存統計結果到檔案供 LLM 分析使用
        agent.save_summary_to_file(summary, "utils/Data/total_stat.txt")
        
        print(f"\n[✓] 異常檢測完成！處理了 {summary['total_rows']} 行資料")
        state['anomaly_checked'] = True
        state['summary_saved'] = True  # 記錄統計檔案已保存
        
    except Exception as e:
        print(f"[!] 異常檢測失敗: {e}")
        state['anomaly_checked'] = False
    
    return state

def llm_summary_analysis(state: WorkflowState, config=None):
    """使用 LLM 分析統計檔案並產生總結"""
    print("\n[Agentic Workflow] 🤖 LLM 正在分析檢測結果...")
    
    try:
        # 讀取統計檔案
        stat_file_path = "utils/Data/total_stat.txt"
        if not os.path.exists(stat_file_path):
            print("[!] 統計檔案不存在，跳過 LLM 分析")
            state['llm_summary_generated'] = False
            return state
        
        with open(stat_file_path, 'r', encoding='utf-8') as f:
            stat_content = f.read()
        
        # 建立分析 prompt
        try:
            analysis_prompt = f"""請作為工業感測器異常檢測專家，分析以下檢測統計報告，並提供專業的總結和建議。

檢測統計報告內容：
{stat_content}

請根據以上統計結果，提供以下分析：

1. **系統整體健康狀況評估** (優秀/良好/一般/需注意/嚴重)
2. **主要問題識別** (列出最嚴重的 2-3 個問題)
3. **根本原因分析** (可能的原因推測)
4. **具體行動建議** (優先級排序的修復步驟)
5. **預防措施建議** (避免未來再次發生)

請用清楚易懂的中文回答，針對設備維護人員提供實用的見解。回答要簡潔明了，重點突出。"""

            # 呼叫統一的 LLM 函數
            llm_summary = Use_LLM(analysis_prompt, temperature=0.7)
            
            if llm_summary:  # 確認 LLM 有回應
                print("\n" + "="*80)
                print("🤖 LLM 智能分析總結")
                print("="*80)
                print(llm_summary)
                print("="*80)
                state['llm_summary_generated'] = True
            else:
                raise Exception("LLM 沒有回應")
            
        except Exception as e:
            print(f"[!] LLM 分析失敗: {e}")
            print("📊 將顯示基本統計摘要")
            
            # LLM 失敗時的簡單摘要
            print("\n" + "="*60)
            print("📊 檢測結果基本摘要")
            print("="*60)
            print(stat_content)
            print("="*60)
            
            state['llm_summary_generated'] = False
    
    except Exception as e:
        print(f"[!] 讀取統計檔案失敗: {e}")
        state['llm_summary_generated'] = False
    
    return state

def user_query(state: WorkflowState, config=None):
    """讓使用者詢問關於數據的問題"""
    print("\n" + "="*80)
    print("📊 數據查詢助手")
    print("="*80)
    print("您可以詢問關於 utils/Data/testing.csv 的任何問題，例如：")
    print("- 數據總共有幾行？")
    print("- 溫度的平均值是多少？")
    print("- 有多少個異常數據？")
    print("- 輸入 'quit' 或 '結束' 退出查詢模式")
    print("="*80)
    
    # 獲取用戶問題
    user_question = input("\n🤔 請輸入您的問題: ").strip()
    
    if user_question.lower() in ['quit', '結束', 'exit', '退出']:
        state['query_completed'] = True
        state['continue_asking'] = False
        print("👋 感謝使用數據查詢助手！")
        return state
    
    state['user_question'] = user_question
    state['query_completed'] = False
    
    # 使用 LLM 判斷是否需要生成代碼
    try:
        # 建立判斷 prompt
        judge_prompt = f"""你是一個數據分析專家。請判斷以下用戶問題是否需要生成 pandas 代碼來回答。

用戶問題："{user_question}"

數據檔案：utils/Data/testing.csv (包含欄位：timestamp, temp, pressure, vibration, label)

請按照以下格式回答：
Code_flag: <yes/no>
Response: <如果不需要代碼，請直接回答問題；如果需要代碼，請說明將要執行什麼分析>

判斷原則：
- 如果問題需要計算、統計、繪圖、數據分析，則回答 yes
- 如果問題是概念性問題、一般性說明，則回答 no 並直接解答

範例：
問題："數據有幾行？" → Code_flag: yes
問題："什麼是溫度感測器？" → Code_flag: no
"""

        # 使用統一的 LLM 函數
        llm_output = Use_LLM(judge_prompt, temperature=0.3)
        
        # 解析 LLM 輸出
        import re
        code_flag_match = re.search(r'Code_flag:\s*(yes|no)', llm_output, re.IGNORECASE)
        response_match = re.search(r'Response:\s*(.*)', llm_output, re.DOTALL)
        
        if code_flag_match:
            state['code_flag'] = code_flag_match.group(1).lower()
        else:
            state['code_flag'] = 'yes'  # 預設需要代碼
        
        if response_match:
            state['llm_code_response'] = response_match.group(1).strip()
        else:
            state['llm_code_response'] = "我將為您分析這個問題。"
        
        print(f"\n🤖 分析結果：{state['llm_code_response']}")
        
    except Exception as e:
        print(f"[!] LLM 判斷失敗: {e}")
        state['code_flag'] = 'yes'  # 預設需要代碼
        state['llm_code_response'] = "我將嘗試用代碼來回答您的問題。"
    
    return state

def code_exe(state: WorkflowState, config=None):
    """執行 LLM 生成的代碼或直接回答問題"""
    
    if state['code_flag'] == 'no':
        # 不需要代碼，直接顯示回答
        print("="*80)
        print(f"💡 回答：{state['llm_code_response']}")
        print("="*80)
    else:
        # 需要生成和執行代碼
        print("\n🔧 正在生成分析代碼...")
        
        try:
            # 建立代碼生成 prompt
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                if retry_count == 0:
                    code_prompt = f"""你是一個 pandas 數據分析專家。請為以下問題生成 Python 代碼。

用戶問題："{state['user_question']}"
數據檔案：utils/Data/testing.csv

數據格式：
- timestamp: 時間戳記
- temp: 溫度值 (可能有空值)
- pressure: 壓力值 (可能有空值)  
- vibration: 振動值 (可能有空值)
- label: 標籤 (normal/abnormal)

請生成簡潔的 Python 代碼，包含：
1. 導入必要的庫 (pandas, numpy, matplotlib.pyplot as plt)
2. 讀取數據: df = pd.read_csv('utils/Data/testing.csv')
3. 分析數據並回答問題
4. 如果是繪圖，請使用 matplotlib 並保存為 PNG 檔案
5. 確保代碼具有錯誤處理 (使用 try-except)
6. **重要：在最後一定要用 print() 輸出結果，讓用戶能看到答案**

請注意：
- 不要使用 `if __name__ == '__main__':` 條件
- 不要定義函數，直接寫執行代碼
- 代碼應該直接執行並產生輸出

範例格式：
```python
import pandas as pd
df = pd.read_csv('utils/Data/testing.csv')
# 分析代碼...
print(f"結果: {{value}}")
```

請只回傳可執行的 Python 代碼，不要包含額外說明："""
                else:
                    # 重試時包含錯誤訊息
                    code_prompt = f"""前一次代碼執行失敗，錯誤訊息：
{state.get('code_error', '未知錯誤')}

請修正代碼並重新生成。

原始問題："{state['user_question']}"
數據檔案：utils/Data/testing.csv

請生成修正後的完整 Python 代碼："""

                # 使用統一的 LLM 函數
                generated_code = Use_LLM(code_prompt, temperature=0.3).strip()
                
                # 清理代碼（移除 markdown 格式）
                import re
                code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
                if code_match:
                    generated_code = code_match.group(1)
                elif '```' in generated_code:
                    # 處理其他 markdown 格式
                    generated_code = re.sub(r'```.*?\n', '', generated_code)
                    generated_code = re.sub(r'\n```', '', generated_code)
                
                print("="*80)
                print("🔧 生成的代碼：")
                print("="*80)
                print(generated_code)
                print("="*80)
                
                # 執行代碼
                try:
                    print("⚡ 執行中...")
                    print("="*80)
                    
                    # 建立執行環境
                    import sys
                    import io
                    
                    # 捕獲標準輸出
                    old_stdout = sys.stdout
                    captured_output = io.StringIO()
                    
                    try:
                        # 建立執行環境，包含完整的全局變數
                        exec_globals = {
                            '__builtins__': __builtins__,
                            '__name__': '__main__',  # 設定 __name__ 為 '__main__'
                            'pandas': __import__('pandas'),
                            'numpy': __import__('numpy'),
                            'matplotlib': __import__('matplotlib'),
                            'plt': __import__('matplotlib.pyplot'),
                            'print': print,  # 確保 print 函數可用
                            'sys': sys,
                            'os': __import__('os'),
                        }
                        
                        # 設定 matplotlib 後端
                        exec_globals['matplotlib'].use('Agg')
                        
                        # 重定向輸出以捕獲結果
                        sys.stdout = captured_output
                        
                        # 執行代碼
                        exec(generated_code, exec_globals)
                        
                        # 恢復標準輸出
                        sys.stdout = old_stdout
                        
                        # 獲取捕獲的輸出
                        output_content = captured_output.getvalue()
                        
                        # 顯示執行結果
                        if output_content.strip():
                            print("📊 執行結果：")
                            print(output_content)
                        else:
                            print("✅ 代碼執行完成（無輸出）")
                        
                    finally:
                        # 確保恢復標準輸出
                        sys.stdout = old_stdout
                        captured_output.close()
                    
                    print("="*80)
                    print("✅ 代碼執行成功！")
                    state['code_execution_result'] = "執行成功"
                    state['code_error'] = ""
                    break
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    print(f"❌ 代碼執行失敗 (第 {retry_count} 次): {error_msg}")
                    state['code_error'] = error_msg
                    
                    if retry_count >= max_retries:
                        print("⚠️ 已達到最大重試次數，無法執行代碼")
                        state['code_execution_result'] = f"執行失敗: {error_msg}"
                        break
                    else:
                        print(f"🔄 正在重試... ({retry_count + 1}/{max_retries})")
        
        except Exception as e:
            print(f"[!] 代碼生成失敗: {e}")
            state['code_execution_result'] = f"生成失敗: {e}"
    
    # 詢問是否繼續
    print("\n" + "="*80)
    continue_choice = input("❓ 是否要繼續詢問其他問題？(y/n): ").strip().lower()
    
    if continue_choice in ['y', 'yes', '是', '繼續']:
        state['continue_asking'] = True
        state['query_completed'] = False
    else:
        state['continue_asking'] = False
        state['query_completed'] = True
        print("👋 感謝使用數據查詢助手！")
    
    return state

def cleanup_files(state: WorkflowState, config=None):
    """清理暫存檔案"""
    try:
        stat_file_path = "utils/Data/total_stat.txt"
        if os.path.exists(stat_file_path):
            os.remove(stat_file_path)
    except Exception as e:
        print(f"[!] 清理檔案失敗: {e}")
    
    return state

def end_node(state: WorkflowState, config=None):
    print("\n[✓] Agentic Workflow 完成！\n")
    return state

def build_workflow():
    from langgraph.graph import StateGraph, END
    
    # Define a new graph with our state
    workflow = StateGraph(WorkflowState)
    
    # 1. Add our nodes 
    workflow.add_node("check_file", check_file)
    workflow.add_node("prompt_params", prompt_params)
    workflow.add_node("user_intent_analysis", user_intent_analysis)
    workflow.add_node("generate_data", generate_data)
    workflow.add_node("anomaly_detection", anomaly_detection)
    workflow.add_node("llm_summary_analysis", llm_summary_analysis)
    workflow.add_node("user_query", user_query)
    workflow.add_node("code_exe", code_exe)
    workflow.add_node("cleanup_files", cleanup_files)
    workflow.add_node("finish", end_node)
    
    # 2. Set the entrypoint as `check_file`, this is the first node called
    workflow.set_entry_point("check_file")
    
    # 3. Add a conditional edge after the `check_file` node is called.
    def check_file_branch(state):
        return "anomaly_detection" if state["file_exists"] else "prompt_params"
    
    workflow.add_conditional_edges(
        "check_file",
        check_file_branch,
        {
            "anomaly_detection": "anomaly_detection",
            "prompt_params": "prompt_params",
        },
    )
    
    # 4. Add conditional edge after prompt_params
    def prompt_params_branch(state):
        return "generate_data" if state["params_confirmed"] else "user_intent_analysis"
    
    workflow.add_conditional_edges(
        "prompt_params",
        prompt_params_branch,
        {
            "generate_data": "generate_data",
            "user_intent_analysis": "user_intent_analysis",
        },
    )
    
    # 5. Add conditional edge after user_intent_analysis
    def intent_analysis_branch(state):
        return "generate_data" if state["params_confirmed"] else "user_intent_analysis"
    
    workflow.add_conditional_edges(
        "user_intent_analysis",
        intent_analysis_branch,
        {
            "generate_data": "generate_data",
            "user_intent_analysis": "user_intent_analysis",  # 可以循環回自己
        },
    )
    
    # 6. Add normal edges after other nodes are called
    workflow.add_edge("generate_data", "anomaly_detection")
    workflow.add_edge("anomaly_detection", "llm_summary_analysis")
    workflow.add_edge("llm_summary_analysis", "user_query")
    
    # 7. Add conditional edge after code_exe
    def code_exe_branch(state):
        return "user_query" if state.get("continue_asking", False) else "cleanup_files"
    
    workflow.add_conditional_edges(
        "code_exe",
        code_exe_branch,
        {
            "user_query": "user_query",
            "cleanup_files": "cleanup_files",
        },
    )
    
    # 6. Add conditional edge after user_query
    def user_query_branch(state):
        return "cleanup_files" if state.get("query_completed", False) else "code_exe"
    
    workflow.add_conditional_edges(
        "user_query",
        user_query_branch,
        {
            "code_exe": "code_exe",
            "cleanup_files": "cleanup_files",
        },
    )
    workflow.add_edge("cleanup_files", "finish")
    
    # Now we can compile and visualize our graph
    return workflow.compile()

def save_workflow_diagram():
    """生成並保存工作流程圖"""
    import os
    import subprocess
    import base64
    import requests
    from PIL import Image, ImageDraw, ImageFont
    
    print("\n🎨 正在生成工作流程圖...")
    
    # 建立工作流程
    app = build_workflow()
    
    try:
        # 嘗試使用 LangGraph 內建的 mermaid 功能
        mermaid_code = app.get_graph(xray=True).draw_mermaid()
        print("✅ 成功生成 Mermaid 代碼")
        
        # 多層次的保存策略
        saved = False
        
        if not saved:
            try:
                print("🌐 嘗試使用 mermaid.ink API...")
                
                # 編碼 mermaid 代碼
                encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('ascii')
                
                # 請求 API
                url = f"https://mermaid.ink/img/{encoded}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    os.makedirs('img', exist_ok=True)
                    with open('img/workflow.png', 'wb') as f:
                        f.write(response.content)
                    print("✅ 成功使用 mermaid.ink API 生成圖片")
                    saved = True
                else:
                    print(f"⚠️ mermaid.ink API 失敗: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ mermaid.ink API 錯誤: {e}")
        
                
            except Exception as e:
                print(f"⚠️ 簡單圖表生成失敗: {e}")
        
        # 保存 mermaid 代碼到檔案
        try:
            os.makedirs('img', exist_ok=True)
            with open('img/workflow.mmd', 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            print("💾 Mermaid 代碼已保存到 img/workflow.mmd")
        except Exception as e:
            print(f"⚠️ 保存 Mermaid 代碼失敗: {e}")
        
        if saved:
            print("🎉 工作流程圖已成功保存到 img/workflow.png")
            print("📄 Mermaid 代碼已保存到 img/workflow.mmd")
        else:
            print("❌ 無法生成工作流程圖，但 Mermaid 代碼已保存")
            
    except Exception as e:
        print(f"❌ 生成工作流程圖失敗: {e}")
        print("請檢查是否已安裝必要的套件")

def main():
    print("=== Agentic AI Workflow 檢查測試資料 ===")
    if '-p' in sys.argv:
        save_workflow_diagram()
        return
    
    # 初始化 state - 因為 TypedDict 不能直接實例化，使用 dict
    state = {
        'num_rows': 300,
        'normal_prob': 0.95,
        'abnormal_prob': 0.3,
        'null_prob': 0.05,
        'file_exists': False,
        'data_path': '',
        'data_generated': False,
        'output': '',
        'anomaly_checked': False,
        'user_input': '',
        'llm_response': '',
        'params_confirmed': False,
        'conversation_active': False,
        'summary_saved': False,
        'llm_summary_generated': False,
        'user_question': '',
        'code_flag': '',
        'llm_code_response': '',
        'code_execution_result': '',
        'code_error': '',
        'query_completed': False,
        'continue_asking': False
    }
    
    app = build_workflow()
    app.invoke(state)

if __name__ == "__main__":
    main()