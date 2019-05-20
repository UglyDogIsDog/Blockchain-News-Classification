# -*- coding:utf-8 -*-

import os
from bert import modeling
import tensorflow as tf
from bert import tokenization

SEN_LEN = 128

flags = tf.flags
FLAGS = flags.FLAGS

bert_path = r'chinese_L-12_H-768_A-12'
root_path = os.getcwd()

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)
flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)
flags.DEFINE_integer(
    "max_seq_length", SEN_LEN,
    "The maximum total input sequence length after WordPiece tokenization."
)

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

def data_preprocess(sentence):
    tokens = []
    for i, word in enumerate(sentence):
        # 分词，如果是中文，就是分字
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    # 序列截断
    if len(tokens) >= FLAGS.max_seq_length - 1:
        tokens = tokens[0:(FLAGS.max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    # print(input_ids)
    input_mask = [1] * len(input_ids)
    # print(input_mask)
    while len(input_ids) < FLAGS.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
    input_ids = [input_ids]
    return input_ids, input_mask

class BertEncoder(object):

    def __init__(self):
        self.bert_model = modeling.BertModel(config=bert_config, is_training=False, max_seq_length=FLAGS.max_seq_length)
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def encode(self, sentence):
        input_ids, input_mask = data_preprocess(sentence)
        return self.sess.run(self.bert_model.embedding_output, feed_dict={self.bert_model.input_ids:input_ids})



if __name__ == "__main__":
    be = BertEncoder()
    embedding = be.encode("据朝日新闻消息，加密货币交易所Zaif之前发生了约70亿日元的加密货币被盗事件，黑客在汇款时连接到互联网的IP地址已被白帽黑客锁定。调查结果显示，IP地址尚未被伪装和匿名，这些与罪犯有关的重要线索目前已经提交给日本金融厅和警察厅。该地址是在欧洲的两个服务器，如果警察当局继续查询，就有可能公开相关信息。币圈屡遭黑客碾压纵观多起虚拟货币交易黑客事件，从分裂以太坊，门头沟事件（2014年2月，黑客从Mt.Gox盗取用户近75万枚比特币及交易所10万枚比特币,直接导致彼时世界第一大交易所Mt.Gox申请破产)，BTER失窃……似乎每一次币圈事故背后都有“黑客”的身影出没。对此，相关人士表示：“古典互联网”黑客正在往币圈大规模迁徙，其仍然用着传统互联网的方法在币圈兴风作浪。古典互联网黑客转行币圈，根本不需要学习成本，而区块链去中心化、匿名的特点和技术手段，导致黑客的行踪很难被追溯，加之法律监管的缺失，更让这群币圈黑客肆意妄为。无论是矿池、钱包、交易所还是公链，甚至是用户的打印机、摄像头，都有被黑客袭击的可能，币圈每年上亿美金的虚拟货币流入黑客口袋。一般来说，币圈黑客攻击分为两种：一种是链上攻击，例如像BTG双花攻击。技术门槛高，攻击者对区块链技术有一定研究；一种是链下服务攻击，比如对交易所、钱包的攻击。意识安全比技术安全更重要世界上有三种人：一种是被黑过，一种是不知道自己被黑过，还有一种是不承认自己被黑过，而交易所就是第三种人。相关人士透露：目前，绝大多数交易所被攻击后，常常装作维护状态，其实是“打破牙齿往肚子里吞”。2014年，BTER交易所发生失窃事件，源于BTERCEO韩林被黑客分析，而其个人密码恰好是BTER交易所里很关键的密码。由此可见，“不论你各个层面的安全防御技术做得再好，如果你人的防御意识出现问题，所有防御都是泡汤的。如今，多数交易所深陷，一边要面对不断涌入币圈的黑客，掌握最顶级的资源，使用最豪华的设备，一边反观自己势单力薄的团队，苦苦挣扎却不被重视的境地。币圈的黑客事件更真实反应出，每个行业的兴起，安全都不会得到重视。被黑客“教育”很多次后，行业才会重视。所以，这不仅仅是技术的更新，更是意识的迭代。币圈黑帽与白帽较量早已开始在Zaif事件中，我们再一次看到了白帽黑客的身影。所谓白帽，亦称白帽黑客、白帽子黑客，是指那些专门研究或者从事网络、计算机技术防御的人，他们通常受雇于各大公司，是维护世界网络、计算机安全的主要力量。很多白帽还受雇于公司，对产品进行模拟黑客攻击，以检测产品的可靠性。因为白帽子只有把自己设身为黑客，去模拟攻击，才能发现漏洞。“以攻促防，只有了解攻击者的手法与心理还有这个群体的生存模式，才能真正做好防御，但充满悖谬的是，监守自盗在事情经常在安全领域上演。有黑客伪装白帽子，获取内部信息后发起攻击。自今年年初，开始做区块链或者转型区块链的安全公司多了起来，币圈白帽子势力正在扩张......对此，业内人士表示：区块链行业里的白帽子非常缺乏，因为安全其本身还是一个服务性的东西，跟黑帽的利益驱动相比，白帽更多是发自内心的责任感去做。但随着行业的极速发展，白帽坚守者会越来越多，它们是黑客精神的铁杆拥护者，为了自身价值开启攻防之战。而在隐秘的币圈战场上，白帽子和黑帽子的较量早就开始了,他们看不见对方，只能在一次次过招时才能感受到对方的存在......声明：本文系九个亿平台原创文章，仅代表作者本人观点。网站、APP等互联网平台用于长期商业目的内容转载须同九个亿签订《内容合作协议》，九个亿保留对各平台内容侵权之一切法律追诉权，“九个亿”所刊载原创内容之知识产权均为“九个亿”所有，欢迎各方转载转发！")
    print(embedding)
    print(embedding.shape)