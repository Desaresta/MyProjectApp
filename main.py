from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen

# For mobile phone
#from android.storage import primary_external_storage_path
#primary_ext_storage = primary_external_storage_path()

from kivy.animation import Animation
from kivy.metrics import dp
from kivymd.toast import toast
from kivy.clock import Clock
from img_process import process, mean_sd_rsd, plot_graph
from kivymd.uix.filemanager import MDFileManager
from kivy.app import App
from kivy.uix.button import ButtonBehavior
from kivy.uix.image import Image
from kivy.uix.popup import Popup
import os


#Window.size = (375, 650)


class ScreenManagement(ScreenManager):
    pass

class HomeScreen(Screen):
    snackbar = None
    _interval = 0

    def show_snackbar(self, snack_type):
        """Create and show instance Snackbar."""

        def callback(instance):
            toast(instance.text)

        def wait_interval(interval):
            self._interval += interval
            if self._interval > self.snackbar.duration + 0.5:
                anim = Animation(y=dp(10), d=0.2)
                anim.start(self.ids.button)
                Clock.unschedule(wait_interval)
                self._interval = 0
                self.snackbar = None

        from kivymd.uix.snackbar import Snackbar
        if snack_type == "warning":
            Snackbar(text="You do not fill out all the information.").open()

    def input_lab_information(self):
        labname = 'ชื่อการทดลอง: ' + self.ids.name.text
        chemi_1 = 'สารเคมีที่ใช้ 1: ' + self.ids.chem1.text
        chemi_2 = 'สารเคมีที่ใช้ 2: ' + self.ids.chem2.text
        chemi_3 = 'สารเคมีที่ใช้ 3: ' + self.ids.chem3.text
        chemi_4 = 'สารเคมีที่ใช้ 4: ' + self.ids.chem4.text
        datetime = 'วันที่ทดลอง: ' + self.ids.date.text
        num = 'ครั้งที่: ' + self.ids.no.text

        if (self.ids.name.text == "" or self.ids.chem1.text == "" or
              self.ids.chem2.text == "" or self.ids.chem3.text == "" or
              self.ids.chem4.text == "" or self.ids.date.text == "" or
              self.ids.no.text == ""):
            type = "warning"
            self.show_snackbar(type)
        else:
            textList = [labname, chemi_1, chemi_2, chemi_3, chemi_4, datetime, num]
            f = open("file_labInfo.txt", 'w', encoding="utf-8")
            for line in textList:
                f.write(line)
                f.write("\n")
            f.close()
            self.manager.current = 'cimg'

class ChooseIMGScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            preview=True,
        )

    def file_manager_open(self):
        self.file_manager.show('/')  # for computer
        #self.file_manager.show(primary_ext_storage)  # for mobile phone
        self.manager_open = True

    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''

        self.exit_manager()
        App.get_running_app().pre_picture_source(path)

        toast(path)


    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()

        return True

class PreProcessScreen(Screen):
    def post(self):
        App.get_running_app().post_picture_source()

class PostProcessScreen(Screen):
    pass

class InitialSubstanceScreen(Screen):
    snackbar = None
    _interval = 0

    def show_snackbar(self, snack_type):
        """Create and show instance Snackbar."""

        def callback(instance):
            toast(instance.text)

        def wait_interval(interval):
            self._interval += interval
            if self._interval > self.snackbar.duration + 0.5:
                anim = Animation(y=dp(10), d=0.2)
                anim.start(self.ids.button)
                Clock.unschedule(wait_interval)
                self._interval = 0
                self.snackbar = None

        from kivymd.uix.snackbar import Snackbar
        if snack_type == "warning":
            Snackbar(text="You do not fill out all the information.").open()

    def intial_sub(self,r,g,b):
        if(self.ids.chem1_1.text == "" or self.ids.chem1_2.text == "" or self.ids.chem1_3.text == "" or self.ids.chem1_4.text == "" or
            self.ids.chem1_5.text == "" or self.ids.chem1_6.text == "" or self.ids.chem1_7.text == "" or self.ids.chem1_8.text == "" or
            self.ids.chem2_1.text == "" or self.ids.chem2_2.text == "" or self.ids.chem2_3.text == "" or self.ids.chem2_4.text == "" or
            self.ids.chem2_5.text == "" or self.ids.chem2_6.text == "" or self.ids.chem2_7.text == "" or self.ids.chem2_8.text == "" or
            self.ids.chem3_1.text == "" or self.ids.chem3_2.text == "" or self.ids.chem3_3.text == "" or self.ids.chem3_4.text == "" or
            self.ids.chem3_5.text == "" or self.ids.chem3_6.text == "" or self.ids.chem3_7.text == "" or self.ids.chem3_8.text == "" or
            self.ids.chem4_1.text == "" or self.ids.chem4_2.text == "" or self.ids.chem4_3.text == "" or self.ids.chem4_4.text == "" or
            self.ids.chem4_5.text == "" or self.ids.chem4_6.text == "" or self.ids.chem4_7.text == "" or self.ids.chem4_8.text == ""):
            type = "warning"
            self.show_snackbar(type)
        else:
            chem1 = [float(self.ids.chem1_1.text), float(self.ids.chem1_2.text),float(self.ids.chem1_3.text), float(self.ids.chem1_4.text),
                     float(self.ids.chem1_5.text), float(self.ids.chem1_6.text),float(self.ids.chem1_7.text), float(self.ids.chem1_8.text)]
            chem2 = [float(self.ids.chem2_1.text), float(self.ids.chem2_2.text),float(self.ids.chem2_3.text), float(self.ids.chem2_4.text),
                     float(self.ids.chem2_5.text), float(self.ids.chem2_6.text),float(self.ids.chem2_7.text), float(self.ids.chem2_8.text)]
            chem3 = [float(self.ids.chem3_1.text), float(self.ids.chem3_2.text),float(self.ids.chem3_3.text), float(self.ids.chem3_4.text),
                     float(self.ids.chem3_5.text), float(self.ids.chem3_6.text),float(self.ids.chem3_7.text), float(self.ids.chem3_8.text)]
            chem4 = [float(self.ids.chem4_1.text), float(self.ids.chem4_2.text),float(self.ids.chem4_3.text), float(self.ids.chem4_4.text),
                     float(self.ids.chem4_5.text), float(self.ids.chem4_6.text),float(self.ids.chem4_7.text), float(self.ids.chem4_8.text)]

            r2=[r,g,b]
            f = open("file_path.txt", "r")
            some_path = f.read()
            file1 = some_path.split("\n")
            x = len(file1)
            file2 = file1[x - 1]
            file3 = file2.split(".")
            filepath = file3[0] + ".xlsx"
            mean_sd_rsd(filepath)
            plot_graph(filepath, chem1, chem2, chem3, chem4, r2)

    def graph_pic_source(self):
        App.get_running_app().graph_picture_source()

class PlotGraph4Pic(Screen):
    pass

class Progress(Popup):
    def __init__(self, **kwargs):
        super(Progress, self).__init__(**kwargs)
        # call dismiss_popup in 2 seconds
        Clock.schedule_once(self.dismiss_popup, 4)

    def dismiss_popup(self, dt):
        self.dismiss()

class ImageButton(ButtonBehavior, Image):
    pass

class MainApp(MDApp):
    #for mobile
    #from android.permissions import request_permissions, Permission
    #request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])

    w = open("file_path.txt", "w")
    w.truncate()
    if os.path.isfile("file_path.txt"):
        with open("file_path.txt", "r") as f:
            some_path = f.read()
            if len(some_path) > 0:
                img_source_path = some_path
            else:
                img_source_path = "empty.jpg"
    else:
        img_source_path = "empty.jpg"


    def pre_picture_source(self, path):
        filename = "C:" + path.replace("\\", "/")
        self.root.ids.pic_pre_id.ids.show_pre.source = filename # For computer
        #self.root.ids.pic_pre_id.ids.show_pre.source = path # For mobile phone
        with open("file_path.txt", "w") as f:

            #f.write(path) # For mobile phone
            f.write(filename)  # For computer
        pic_name = path.split("\\")
        #pic_name = path.split("/") # For mobile phone
        self.root.ids.pic_pre_id.ids.file_name_id.text = pic_name[len(pic_name)-1]

    def post_picture_source(self):
        f = open("file_path.txt", "r")
        filename = f.read()
        filename2 = filename.split('.')
        files = filename.split('_')
        ft = filename2[0] + "_detect.jpg"

        self.root.ids.pic_post_id.ids.show_post.source = ft
        with open("file_path.txt", "a") as f:
            if not filename.endswith('\n'):
                f.write('\n')
            f.write(ft)

    def img_detect(self):
        with open("file_path.txt", "r") as f:
            some_path = f.read()
            if len(some_path) > 0:
                process()
            else:
                print("no image")

    def graph_picture_source(self):
        self.root.ids.gpic_id.ids.figure1.source = 'figure1.png'
        self.root.ids.gpic_id.ids.figure2.source = 'figure2.png'
        self.root.ids.gpic_id.ids.figure3.source = 'figure3.png'
        self.root.ids.gpic_id.ids.figure4.source = 'figure4.png'

    def build(self):
        return ScreenManagement()

    def change_screen(self, screen):
        self.root.current = screen

if __name__ == '__main__':
    MainApp().run()
