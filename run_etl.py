{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fe97e544-0c34-4570-8512-4709b32d1c1d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   category                                               text\n",
      "0  WELLNESS  143 Miles in 35 Days: Lessons Learned Resting ...\n",
      "1  WELLNESS  Talking to Yourself: Crazy or Crazy Helpful? T...\n",
      "2  WELLNESS  Crenezumab: Trial Will Gauge Whether Alzheimer...\n",
      "3  WELLNESS  Oh, What a Difference She Made If you want to ...\n",
      "4  WELLNESS  Green Superfoods First, the bad news: Soda bre...\n",
      "Index(['category', 'text'], dtype='object')\n",
      "Train size: 32939\n",
      "Val size: 8235\n",
      "Late data size: 4575\n",
      "   label                                               text\n",
      "0      1  LAW ENFORCEMENT ON HIGH ALERT Following Threat...\n",
      "1      1  UNBELIEVABLE! OBAMA’S ATTORNEY GENERAL SAYS MO...\n",
      "2      0  Bobby Jindal, raised Hindu, uses story of Chri...\n",
      "3      1  SATAN 2: Russia unvelis an image of its terrif...\n",
      "4      1  About Time! Christian Group Sues Amazon and SP...\n",
      "Index(['label', 'text'], dtype='object')\n",
      "Train size: 51506\n",
      "Val size: 12877\n",
      "Late data size: 7154\n",
      "0: Harry. Potter star. Daniel. Radcliffe gets £20M fortune as he turns 18. Monday. Young actor says he has no plans to fritter his cash away. Radcliffe's earnings from first five. Potter films have been held in trust fund.\n",
      "\n",
      "1: Mentally ill inmates in. Miami are housed on the \"forgotten floor\". Judge. Steven. Leifman says most are there as a result of \"avoidable felonies\". While. CNN tours facility, patient shouts: \"I am the son of the president\". Leifman says the system is unjust and he's fighting for change.\n",
      "\n",
      "2: NEW: \"I thought. I was going to die,\" driver says. Man says pickup truck was folded in half; he just has cut on face. Driver: \"I probably had a 30-, 35-foot free fall\". Minnesota bridge collapsed during rush hour. Wednesday.\n",
      "\n",
      "3: Five small polyps found during procedure; \"none worrisome,\" spokesman says. President reclaims powers transferred to vice president. Bush undergoes routine colonoscopy at. Camp. David.\n",
      "\n",
      "4: NEW: NFL chief, Atlanta. Falcons owner critical of. Michael. Vick's conduct. NFL suspends. Falcons quarterback indefinitely without pay. Vick admits funding dogfighting operation but says he did not gamble. Vick due in federal court. Monday; future in. NFL remains uncertain.\n",
      "\n",
      "Train size: 71995\n",
      "Val size: 17999\n",
      "Late data size: 10000\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "from extract import load_csv, load_csv_welfake, load_csv_summary\n",
    "from filter import clean_text, clean_text_welfake, clean_text_summary\n",
    "from trans import normalize_text, normalize_text_welfake, normalize_text_summary\n",
    "from load import save_jsonl\n",
    "from split import split_data\n",
    "\n",
    "def run_etl_classification(output_dir):\n",
    "\n",
    "    df = load_csv()\n",
    "    df = clean_text(df)\n",
    "    df = normalize_text(df)\n",
    "    print(df.head())\n",
    "    print(df.columns)\n",
    "    train, val, late = split_data(df)\n",
    "\n",
    "    os.makedirs(output_dir, exist_ok=True)\n",
    "    save_jsonl(train, f\"{output_dir}/train.jsonl\")\n",
    "    save_jsonl(val, f\"{output_dir}/val.jsonl\")\n",
    "    save_jsonl(late, f\"{output_dir}/late_data.jsonl\")\n",
    "\n",
    "def run_etl_welfake(output_dir):\n",
    "    df = load_csv_welfake()\n",
    "    df = clean_text_welfake(df)\n",
    "    df = normalize_text_welfake(df)\n",
    "    print(df.head())\n",
    "    print(df.columns)\n",
    "    train, val, late = split_data(df)\n",
    "    \n",
    "    os.makedirs(output_dir, exist_ok=True)\n",
    "    save_jsonl(train, f\"{output_dir}/train.jsonl\")\n",
    "    save_jsonl(val, f\"{output_dir}/val.jsonl\")\n",
    "    save_jsonl(late, f\"{output_dir}/late_data.jsonl\")\n",
    "    \n",
    "def run_etl_summary(output_dir):\n",
    "    df = load_csv_summary()\n",
    "    df = clean_text_summary(df)\n",
    "    df = normalize_text_summary(df)\n",
    "    for i, summary in enumerate(df[\"summary\"].head(5)):\n",
    "        print(f\"{i}: {summary}\\n\")\n",
    "\n",
    "    train, val, late = split_data(df)\n",
    "    \n",
    "    os.makedirs(output_dir, exist_ok=True)\n",
    "    save_jsonl(train, f\"{output_dir}/train.jsonl\")\n",
    "    save_jsonl(val, f\"{output_dir}/val.jsonl\")\n",
    "    save_jsonl(late, f\"{output_dir}/late_data.jsonl\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "\n",
    "    run_etl_classification(\"classification\")\n",
    "    run_etl_welfake(\"welfake\")\n",
    "    run_etl_summary(\"summary\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5a60207e-4617-4d88-a0ab-13f607b54ee4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'article': 'LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won\\'t cast a spell on him. Daniel Radcliffe as Harry Potter in \"Harry Potter and the Order of the Phoenix\" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. \"I don\\'t plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar,\" he told an Australian interviewer earlier this month. \"I don\\'t think I\\'ll be particularly extravagant. \"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs.\" At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film \"Hostel: Part II,\" currently six places below his number one movie on the UK box office chart. Details of how he\\'ll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. \"I\\'ll definitely have some sort of party,\" he said in an interview. \"Hopefully none of you will be reading about it.\" Radcliffe\\'s earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. \"People are always looking to say \\'kid star goes off the rails,\\'\" he told reporters last month. \"But I try very hard not to go that way because it would be too easy for them.\" His latest outing as the boy wizard in \"Harry Potter and the Order of the Phoenix\" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films.  Watch I-Reporter give her review of Potter\\'s latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called \"My Boy Jack,\" about author Rudyard Kipling and his son, due for release later this year. He will also appear in \"December Boys,\" an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer\\'s \"Equus.\" Meanwhile, he is braced for even closer media scrutiny now that he\\'s legally an adult: \"I just think I\\'m going to be more sort of fair game,\" he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.', 'highlights': \"Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday .\\nYoung actor says he has no plans to fritter his cash away .\\nRadcliffe's earnings from first five Potter films have been held in trust fund .\", 'id': '42c027e4ff9730fbb3de84c1af0d2c506e41c3e4'}\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "# 加载训练集\n",
    "dataset = load_dataset(\"cnn_dailymail\", \"3.0.0\", split=\"train\")\n",
    "\n",
    "# 转为 pandas DataFrame\n",
    "df = dataset.to_pandas()\n",
    "\n",
    "# 保存为 CSV 文件（默认用 UTF-8 编码）\n",
    "df.to_csv(\"cnn_dailymail_train.csv\", index=False)\n",
    "# 查看前几条数据\n",
    "print(dataset[0]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4112db33",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "d:\\anaconda\\envs\\MLenv\\lib\\site-packages\\huggingface_hub\\file_download.py:144: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\\Users\\youji\\.cache\\huggingface\\hub\\datasets--EdinburghNLP--xsum. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.\n",
      "To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development\n",
      "  warnings.warn(message)\n",
      "Generating train split: 100%|██████████| 204045/204045 [00:00<00:00, 443393.35 examples/s]\n",
      "Generating validation split: 100%|██████████| 11332/11332 [00:00<00:00, 422371.19 examples/s]\n",
      "Generating test split: 100%|██████████| 11334/11334 [00:00<00:00, 379029.36 examples/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'document': 'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.\\nRepair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.\\nTrains on the west coast mainline face disruption due to damage at the Lamington Viaduct.\\nMany businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.\\nFirst Minister Nicola Sturgeon visited the area to inspect the damage.\\nThe waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.\\nJeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.\\nHowever, she said more preventative work could have been carried out to ensure the retaining wall did not fail.\\n\"It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we\\'re neglected or forgotten,\" she said.\\n\"That may not be true but it is perhaps my perspective over the last few days.\\n\"Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?\"\\nMeanwhile, a flood alert remains in place across the Borders because of the constant rain.\\nPeebles was badly hit by problems, sparking calls to introduce more defences in the area.\\nScottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.\\nThe Labour Party\\'s deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.\\nHe said it was important to get the flood protection plan right but backed calls to speed up the process.\\n\"I was quite taken aback by the amount of damage that has been done,\" he said.\\n\"Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses.\"\\nHe said it was important that \"immediate steps\" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.\\nHave you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.', 'summary': 'Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.', 'id': '35232142'}\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "# 加载训练集\n",
    "dataset = load_dataset(\"EdinburghNLP/xsum\", split=\"train\")\n",
    "\n",
    "# 转为 pandas DataFrame\n",
    "df = dataset.to_pandas()\n",
    "\n",
    "# 保存为 CSV 文件（默认用 UTF-8 编码）\n",
    "df.to_csv(\"xsum_train.csv\", index=False)\n",
    "# 查看前几条数据\n",
    "print(dataset[0]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "95d4cb2e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ 成功生成 summarization_combined.csv\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "import pandas as pd\n",
    "\n",
    "# 加载前 50000 条 CNN/DailyMail 样本\n",
    "cnn = load_dataset(\"cnn_dailymail\", \"3.0.0\", split=\"train[:50000]\").to_pandas()\n",
    "cnn = cnn.rename(columns={\"article\": \"document\", \"highlights\": \"summary\"})\n",
    "cnn = cnn[[\"document\", \"summary\"]]  # 只保留这两列\n",
    "\n",
    "# 加载前 50000 条 XSum 样本\n",
    "xsum = load_dataset(\"EdinburghNLP/xsum\", split=\"train[:50000]\").to_pandas()\n",
    "xsum = xsum[[\"document\", \"summary\"]]  # 只保留这两列\n",
    "\n",
    "# 合并两个数据集\n",
    "combined = pd.concat([cnn, xsum], ignore_index=True)\n",
    "\n",
    "# 保存为 CSV 文件\n",
    "combined.to_csv(\"summarization_combined——new.csv\", index=False)\n",
    "print(\"✅ 成功生成 summarization_combined.csv\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94efbd1a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "MLenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
