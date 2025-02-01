#region ########################### DIANN 2023 #####################  
prompt_t1_diann = """You are an excellent assistant at finding disabilities in Spanish biomedical abstracts. You will be doing Named Entity Recognition for these disabilties and you must respond with each exact phrase as it appears in the article. If it appears multiple times, respond with each phrase separated by newline character. If there are no disabilities, respond with NA.

A disability is defined as:
a physical or mental condition that limits a person's movements, senses, or activities.


Abstract:

"""

#endregion ########################### DIANN 2023 #####################  


#region ########################### DIPROMATS 2023 #####################  
prompt_t1_dipromats = """You are an excellent assistant at identifying propaganda in tweets. Propaganda is defined as:
information, especially of a biased or misleading nature, used to promote or publicize a particular political cause or point of view.

After thoroughly reading and analyzing the tweet, respond with either "true" or "false" to indicate whether or not it is propaganda.

Tweet:

"""
prompt_t2_dipromats = """You are an excellent assistant at categorizing propaganda in tweets. Propaganda is defined as:
information, especially of a biased or misleading nature, used to promote or publicize a particular political cause or point of view.

You will need to decide which of the following applies to the tweet. It could be one or more of the following.

1. Appeal to commonality. This could be related to the following:
  - Ad populum: the tweet appeals to the will, the tradition or the history of a community to support an argument. e.g. "The leadership of the #CPC is the choice of history and of the Chinese people."
  - Flag Waving: the tweet includes hyperbolic praise of a nation, worships a patriotic symbol, exhibits self-praise, or portrays someone as a hero. e.g. "The European Union is the best example, in the history of the world, of conflict resolution."
2. Discrediting the opponent. This could be related to the following:
  - Name Calling/Labelling: the author refers to someone or something with pejorative labels. e.g. "The #US is the gravest threat to global strategic security and stability"
  - Undiplomatic Assertiveness/Whataboutism: the tweet vilifies an opponent, depicting their behavior as hostile, hypocritical or immoral, displaying undiplomatic contempt. This technique also includes counteraccusations to deviate the attention from sensitive issues. e.g. "Just another proof that the #MediaFreedom principle is only applied to western or western-paid media. When Euro-NATO governments crack down on #Russian or Russian-language media there's zero reaction from #HumanRights apologists. Bias and double standards"
  - Scapegoating: the tweet transfers the blame to one person, group or institution. e.g. "What has caused the current difficulties in China-UK relationship? My answer is loud and clear: China has not changed. It is the UK that has changed. The UK side should take full responsibility for the current difficulties."
  - Propaganda Slinging:the author accuse others of spreading propaganda, disinformation or lies. e.g. "Pompeo has been churning out lies wherever he goes, spreading political virus across the world."
  - Personal attacks: the author attacks the personal, private background of an opponent. e.g. "He tries to appeal to Christian voters, but his real life is anything but Christian. He is a heavy drinker and a compulsive womanizer."
  - Fear Appeal: the author either seeks to instill fear in the readers about hypothetical situations that an opponent may provoke or aims to intimidate an opponent by warning about the consequences of their actions. e.g. "We urge the US to stop using the Uighur Human Rights Policy Act of 2020 to harm China's interests. Otherwise, China will resolutely fight back, and the US will bear all the consequences."
  - Absurdity Appeal: the author characterizes the behavior of an opponent or their ideas as absurd, ridiculous or pathetic. e.g. "Joe Biden's response to the H1N1 Swine Flu was pathetic. Joe didnt have a clue!"
  - Demonization: the author invokes civic hatred towards an opponent, who is presented as an existential threat. e.g. "Concast (@NBCNews) and Fake News @CNN are Chinese puppets who want to do business there. They use USA airwaves to help China. The Enemy of the People!"
  - Doubt: The author casts doubt on the credibility or honesty of someone. e.g. "Growing doubts over the US government's handling of the #COVID19, e.g. When did the first infection occur in the US? Is the US government hiding something? Why they opt to blame others?"
  - Reductio ad Hitlerum: the tweets try to persuade an audience to disapprove an action or idea from an opponent by associating it with someone or something that is hated by the audience. e.g. "The CPC has 90 million members, plus their families, the data has at least 270 million. Infringing these elites is directly against the Chinese people. Don't forget Hitler's evil history of persecution and massacres of German Communists and Jews.Stop NEW horrible fascists!"
3. Loaded Language. This mainly concerns hyperbolic language, evocative metaphors and words with strong emotional connotations. For example: "this monumental achievement left a tremendous mark in history!"
4. Appeal to Authority. 
  - Appeal to false authority: Tweet includes a third person or institution to support an idea, message, or behavior for which they should not be considered as a valid expert. e.g. "A voice of a Pakistani student's wife tells real situation about the coronavirus in China. Trust the Chinese Government. No panic!"
  - Bandwagoning: The author seeks to persuade someone to join a course of action because someone else is doing it. e.g. "Germany took strong action today against Hizballah. We call on #EU member states to follow suit in holding Hizballah accountable."

If none of the four categories above apply, respond "false" to indicate it is not propaganda.

After thoroughly reading and analyzing the tweet, choose one or more of the above categories, and if none of the four applies, choose option 5:

1 appeal to commonality
2 discrediting the opponent
3 loaded language
4 appeal to authority
5 not propaganda

Tweet:

"""
prompt_t3_dipromats = """You are an excellent assistant at categorizing propaganda in tweets. Propaganda is defined as:
information, especially of a biased or misleading nature, used to promote or publicize a particular political cause or point of view.

You will need to decide which of the following applies to the tweet. It could be one or more of the following.

A. Appeal to commonality - Ad populum: the tweet appeals to the will, the tradition or the history of a community to support an argument. e.g. "The leadership of the #CPC is the choice of history and of the Chinese people."
B. Appeal to commonality - Flag Waving: the tweet includes hyperbolic praise of a nation, worships a patriotic symbol, exhibits self-praise, or portrays someone as a hero. e.g. "The European Union is the best example, in the history of the world, of conflict resolution."
C. Discrediting the opponent - Name Calling/Labelling: the author refers to someone or something with pejorative labels. e.g. "The #US is the gravest threat to global strategic security and stability"
D. Discrediting the opponent - Undiplomatic Assertiveness/Whataboutism: the tweet vilifies an opponent, depicting their behavior as hostile, hypocritical or immoral, displaying undiplomatic contempt. This technique also includes counteraccusations to deviate the attention from sensitive issues. e.g. "Just another proof that the #MediaFreedom principle is only applied to western or western-paid media. When Euro-NATO governments crack down on #Russian or Russian-language media there's zero reaction from #HumanRights apologists. Bias and double standards"
E. Discrediting the opponent - Scapegoating: the tweet transfers the blame to one person, group or institution. e.g. "What has caused the current difficulties in China-UK relationship? My answer is loud and clear: China has not changed. It is the UK that has changed. The UK side should take full responsibility for the current difficulties."
F. Discrediting the opponent - Propaganda Slinging:the author accuse others of spreading propaganda, disinformation or lies. e.g. "Pompeo has been churning out lies wherever he goes, spreading political virus across the world."
G. Discrediting the opponent - Personal attacks: the author attacks the personal, private background of an opponent. e.g. "He tries to appeal to Christian voters, but his real life is anything but Christian. He is a heavy drinker and a compulsive womanizer."
H. Discrediting the opponent - Fear Appeals: the author either seeks to instill fear in the readers about hypothetical situations that an opponent may provoke or aims to intimidate an opponent by warning about the consequences of their actions. e.g. "We urge the US to stop using the Uighur Human Rights Policy Act of 2020 to harm China's interests. Otherwise, China will resolutely fight back, and the US will bear all the consequences."
I. Discrediting the opponent - Absurdity Appeal: the author characterizes the behavior of an opponent or their ideas as absurd, ridiculous or pathetic. e.g. "Joe Biden's response to the H1N1 Swine Flu was pathetic. Joe didnt have a clue!"
J. Discrediting the opponent - Demonization: the author invokes civic hatred towards an opponent, who is presented as an existential threat. e.g. "Concast (@NBCNews) and Fake News @CNN are Chinese puppets who want to do business there. They use USA airwaves to help China. The Enemy of the People!"
K. Discrediting the opponent - Doubt: The author casts doubt on the credibility or honesty of someone. e.g. "Growing doubts over the US government's handling of the #COVID19, e.g. When did the first infection occur in the US? Is the US government hiding something? Why they opt to blame others?"
L. Discrediting the opponent - Reductio ad Hitlerum: the tweets try to persuade an audience to disapprove an action or idea from an opponent by associating it with someone or something that is hated by the audience. e.g. "The CPC has 90 million members, plus their families, the data has at least 270 million. Infringing these elites is directly against the Chinese people. Don't forget Hitler's evil history of persecution and massacres of German Communists and Jews.Stop NEW horrible fascists!"
M. Loaded Language. This mainly concerns hyperbolic language, evocative metaphors and words with strong emotional connotations. For example: "this monumental achievement left a tremendous mark in history!"
N. Appeal to Authority - Appeal to false authority: Tweet includes a third person or institution to support an idea, message, or behavior for which they should not be considered as a valid expert. e.g. "A voice of a Pakistani student's wife tells real situation about the coronavirus in China. Trust the Chinese Government. No panic!"
O. Appeal to Authority - Bandwagoning: The author seeks to persuade someone to join a course of action because someone else is doing it. e.g. "Germany took strong action today against Hizballah. We call on #EU member states to follow suit in holding Hizballah accountable."

If none of the four categories above apply, respond "false" to indicate it is not propaganda.

After thoroughly reading and analyzing the tweet, choose one or more of the above categories, unless if none of options A-O applies, choose option P:

A appeal to commonality - ad populum
B appeal to commonality - flag waving
C discrediting the opponent - name calling
D discrediting the opponent - undiplomatic assertiveness/whataboutism
E discrediting the opponent - scapegoating
F discrediting the opponent - propaganda slinging
G discrediting the opponent - personal attacks
H discrediting the opponent - fear appeals
I discrediting the opponent - absurdity appeal
J discrediting the opponent - demonization
K discrediting the opponent - doubt
L discrediting the opponent - reductio ad hitlerum
M loaded language
N appeal to authority - appeal to false authority
O appeal to authority - bandwagoning
P not propaganda

Tweet:

"""

#endregion ########################### DIPROMATS 2023 #####################  


#region ########################### Exist 2022 #####################  

prompt_t1_exist_2022 = """You are an excellent assistant at identifying sexism in spanish tweets. You must decide whether or not a given tweet expresses ideas related to sexism in any of the three forms: it is sexist itself, it describes a sexist situation in which discrimination towards women occurs, or criticizes a sexist behaviour. The following statements show examples of sexist and not sexist messages, respectively.  Sexism is defined as prejudice, stereotyping, or discrimination, typically against women, on the basis of sex.


Sexist tweet: Woman driving, be careful!
Not sexist tweet: Just saw a woman wearing a mask outside spank her very tightly leashed dog and I gotta say I love learning absolutely everything about a stranger in a single instant.

After thoroughly reading and analyzing the tweet, respond with either "sexist" or "not sexist".


Tweet:

"""

t2_exist_labels = [
    "ideological inequality",
    "stereotyping dominance",
    "sexual violence",
    "misogyny and/or non-sexual violence",
    "objectification",
    "not sexist",
]

numbered_t2_exist_labels = [f"{i}. {l}" for i,l in enumerate(t2_exist_labels, start=1)]
numbered_t2_exist_labels_str = "\n".join(numbered_t2_exist_labels)

prompt_t2_exist_2022 = f"""You are an excellent assistant at categorizing sexism in spanish tweets. Sexism is defined as prejudice, stereotyping, or discrimination, typically against women, on the basis of sex.

Here are the category definitions:

1. Ideological and inequality. This category includes all tweets that discredit the feminist movement in order to devalue, belittle and defame the struggle of women in any aspect
of their lives. It also includes messages that reject inequality between men and women, or present men as victims of gender-based oppression. Some examples of this type of sexism discourse can be found in the following tweets:  "#Feminism is a war on men, but it's also a war on women. It's a war on female nature, a war on femininity."
2. Role stereotyping and dominance. This category includes messages that express false ideas about women that suggest they are more suitable or inappropriate for certain tasks. It also includes any claim that implies that men are somehow superior to women. "Most women no longer have the desire or the knowledge to develop a high quality character, even if they wanted to."
3. Objectification. Objectification and physical stereotyping includes messages where women are presented as objects apart from their dignity and personal aspects. We also include messages that assume or describe certain physical qualities that women must have in order to fulfill traditional gender roles, for example, ideas that suggest that women should maintain a standard and ideal of beauty or attacks on a woman's physique. "No offense but I've never seen an attractive african american hooker. Not a single one"
4. Sexual violence. This category includes messages where sexual suggestions, requests or harassment of a sexual nature (rape or sexual assault) are made: "I wanna touch your tits..you can't imagine what I can do on your body"
5. Misogyny and non sexual violence: this category includes expressions of hatred and violence towards women: "Domestic abuse is never okay.... Unless your wife is a bitch."
6. Not sexist. No sexism present in the text.

For this task, you will be shown a tweet in Spanish. After thoroughly reading and analyzing the tweet, categorize the tweet into ONLY ONE of the six categories by using the following labels:

{numbered_t2_exist_labels_str}


Tweet:

"""

#endregion ########################### Exist 2022 #####################  


#region ########################### Exist 2023 #####################  

# same task, same prompt
prompt_t1_exist_2023 = prompt_t1_exist_2022

prompt_t2_exist_2023 = """""You are an excellent assistant at identifying the source in sexist spanish tweets. Sexism is defined as prejudice, stereotyping, or discrimination, typically against women, on the basis of sex. 

Category definitions:

1. Direct sexist message. The intention was to write a message that is sexist by itself or incites to be sexist, as in: "Women shouldn't code...perhaps be influencer/creator instead...it's their natural
strength."
2. Reported sexist message. The intention is to report and share a sexist situation suffered by a woman or women in first or third person, as in: "Today, one of my year 1 class pupils could not believe he'd lost a race against a girl."
3. Judgemental message. The intention was judgmental, since the tweet describes sexist situations or behaviours with the aim of condemning them. As in: "21st century and we are still earning 25% less than men #Idonotrenounce."
4. Not sexist message. No sexism present in the text.

For this task, you will be shown a tweet in Spanish. After thoroughly reading and analyzing the tweet, categorize the tweet into one of the above categories using the following labels:

1. direct
2. reported
3. judgmental
4. not sexist


Tweet:

"""

prompt_t3_exist_2023 = f"""You are an excellent assistant at categorizing sexism in spanish tweets. Sexism is defined as prejudice, stereotyping, or discrimination, typically against women, on the basis of sex.

Here are the category definitions:

1. Ideological and inequality. This category includes all tweets that discredit the feminist movement in order to devalue, belittle and defame the struggle of women in any aspect
of their lives. It also includes messages that reject inequality between men and women, or present men as victims of gender-based oppression. Some examples of this type of sexism discourse can be found in the following tweets:  "#Feminism is a war on men, but it's also a war on women. It's a war on female nature, a war on femininity."
2. Role stereotyping and dominance. This category includes messages that express false ideas about women that suggest they are more suitable or inappropriate for certain tasks. It also includes any claim that implies that men are somehow superior to women. "Most women no longer have the desire or the knowledge to develop a high quality character, even if they wanted to."
3. Objectification. Objectification and physical stereotyping includes messages where women are presented as objects apart from their dignity and personal aspects. We also include messages that assume or describe certain physical qualities that women must have in order to fulfill traditional gender roles, for example, ideas that suggest that women should maintain a standard and ideal of beauty or attacks on a woman's physique. "No offense but I've never seen an attractive african american hooker. Not a single one"
4. Sexual violence. This category includes messages where sexual suggestions, requests or harassment of a sexual nature (rape or sexual assault) are made: "I wanna touch your tits..you can't imagine what I can do on your body"
5. Misogyny and non sexual violence: this category includes expressions of hatred and violence towards women: "Domestic abuse is never okay.... Unless your wife is a bitch."
6. Not sexist. No sexism present in the text.

For this task, you will be shown a tweet in Spanish. After thoroughly reading and analyzing the tweet, categorize the tweet into ONE OR MORE of the above six categories using the following labels:

{numbered_t2_exist_labels_str}


Tweet:

"""

#endregion ########################### Exist 2023 #####################  


#region ########################### SQAC #####################  


prompt_t1_sqac = """You are an excellent assistant at answering questions based on Spanish texts. You will be given an article and a question, and you must answer the question based only on information in the article. The response should be extracted from the context verbatim.

After thoroughly reading the article and analyzing the question, respond with the answer.

Title:
{title}

Context:
{context}

Question:
{question}
"""
#endregion ########################### SQAC #####################  